from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import config


def _configure_ssl_bundle_path() -> None:
    try:
        import certifi

        src_ca = certifi.where()
        if not src_ca or not os.path.exists(src_ca):
            return

        ascii_ca = os.path.join(tempfile.gettempdir(), "yfinance-cacert.pem")
        if (not os.path.exists(ascii_ca)) or (os.path.getsize(ascii_ca) != os.path.getsize(src_ca)):
            shutil.copyfile(src_ca, ascii_ca)

        os.environ.setdefault("SSL_CERT_FILE", ascii_ca)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ascii_ca)
        os.environ.setdefault("CURL_CA_BUNDLE", ascii_ca)
    except Exception:
        return


_configure_ssl_bundle_path()

try:
    import yfinance as yf
except ModuleNotFoundError:
    yf = None


st.set_page_config(page_title="Market Direction Dashboard", page_icon="MD", layout="wide")

ASSET_OPTIONS = {
    "S&P 500": "sp500",
    "Nasdaq": "nasdaq",
    "Gold": "gold",
    "Silver": "silver",
}

PERIOD_OPTIONS = ["1H", "4H", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
PERIOD_LABELS = {
    "1H": "hour",
    "4H": "4-hour block",
    "Daily": "day",
    "Weekly": "week",
    "Monthly": "month",
    "Quarterly": "quarter",
    "Yearly": "year",
}


def _asset_settings(asset_key: str) -> dict[str, str]:
    return config.ASSET_SETTINGS[asset_key]


def _theme_mode() -> str:
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "dark"
    return st.session_state.theme_mode


def _toggle_theme() -> None:
    st.session_state.theme_mode = "light" if _theme_mode() == "dark" else "dark"


def _theme_palette() -> dict[str, str]:
    if _theme_mode() == "light":
        return {
            "bg": "#f6f8fc",
            "panel": "#ffffff",
            "text": "#0f172a",
            "muted": "#475569",
            "border": "#dbe4f0",
            "accent": "#0284c7",
            "accent2": "#059669",
            "plotly": "plotly_white",
        }
    return {
        "bg": "#071021",
        "panel": "#0d1a33",
        "text": "#e7efff",
        "muted": "#94a3b8",
        "border": "rgba(255,255,255,0.10)",
        "accent": "#67e8f9",
        "accent2": "#34d399",
        "plotly": "plotly_dark",
    }


def _inject_css() -> None:
    t = _theme_palette()
    st.markdown(
        f"""
        <style>
            .stApp {{ background: {t['bg']}; color: {t['text']}; }}
            [data-testid='stSidebar'] {{ background: {t['panel']}; border-right: 1px solid {t['border']}; }}
            .topbar {{ background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 14px; padding: 0.7rem 1rem; margin-bottom: 0.9rem; }}
            .hero {{ background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 14px; padding: 1rem 1.1rem; margin-bottom: 0.8rem; }}
            .hero h1 {{ margin: 0; font-size: 1.9rem; }}
            .hero p {{ margin: 0.4rem 0 0; color: {t['muted']}; }}
            .metric-card {{ background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 12px; padding: 0.8rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def add_features(df: pd.DataFrame, ma_short: int, ma_long: int, vol_window: int, rsi_window: int, bb_window: int) -> pd.DataFrame:
    d = df.copy()
    d["Direction"] = (d["Close"] > d["Open"]).astype(int)
    d["LogReturn"] = np.log(d["Close"] / d["Close"].shift(1))

    d["MA_short"] = d["Close"].rolling(ma_short).mean()
    d["MA_long"] = d["Close"].rolling(ma_long).mean()
    d["MA_cross"] = d["MA_short"] - d["MA_long"]

    d["Volatility"] = d["LogReturn"].rolling(vol_window).std()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_window).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_window).mean()
    rs = gain / (loss + 1e-12)
    d["RSI"] = 100 - (100 / (1 + rs))

    bb_mid = d["Close"].rolling(bb_window).mean()
    bb_std = d["Close"].rolling(bb_window).std()
    d["BB_upper"] = bb_mid + config.BB_STD * bb_std
    d["BB_lower"] = bb_mid - config.BB_STD * bb_std
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / bb_mid
    d["BB_pct"] = (d["Close"] - d["BB_lower"]) / (d["BB_upper"] - d["BB_lower"])

    return d


def scaled_windows(divisor: float, min_w: int = 2) -> tuple[int, int, int, int, int]:
    ma_short = max(min_w, int(round(config.MA_SHORT / divisor)))
    ma_long = max(ma_short + 1, int(round(config.MA_LONG / divisor)))
    vol_w = max(min_w, int(round(config.VOL_WINDOW / divisor)))
    rsi_w = max(min_w, int(round(config.RSI_WINDOW / divisor)))
    bb_w = max(min_w, int(round(config.BB_WINDOW / divisor)))
    return ma_short, ma_long, vol_w, rsi_w, bb_w


def intraday_windows(hours_per_bar: int) -> tuple[int, int, int, int, int]:
    bars_per_day = max(1, int(round(6.5 / hours_per_bar)))
    ma_short = max(6, config.MA_SHORT * bars_per_day)
    ma_long = max(ma_short + 2, config.MA_LONG * bars_per_day)
    vol_w = max(6, config.VOL_WINDOW * bars_per_day)
    rsi_w = max(6, config.RSI_WINDOW * bars_per_day)
    bb_w = max(6, config.BB_WINDOW * bars_per_day)
    return ma_short, ma_long, vol_w, rsi_w, bb_w


FEATURES = ["MA_cross", "Volatility", "RSI", "BB_width", "BB_pct", "LogReturn"]


def _fit_model(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[RandomForestClassifier, StandardScaler]:
    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    model = RandomForestClassifier(n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE, n_jobs=-1)
    model.fit(x_train_sc, y_train)
    return model, scaler


def _predict_from_frame(frame: pd.DataFrame, train_before: pd.Timestamp, target_date: pd.Timestamp, min_rows: int, windows: tuple[int, int, int, int, int]) -> dict:
    ma_s, ma_l, vol_w, rsi_w, bb_w = windows
    feat = add_features(frame, ma_s, ma_l, vol_w, rsi_w, bb_w)

    x = feat[FEATURES].shift(1)
    y = feat["Direction"]

    train_mask = x.notna().all(axis=1) & y.notna() & (x.index < train_before)
    x_train = x.loc[train_mask]
    y_train = y.loc[train_mask].astype(int)
    if len(x_train) < min_rows:
        raise ValueError(f"Not enough training rows: {len(x_train)}")

    model, scaler = _fit_model(x_train, y_train)

    pred_date = target_date if target_date in x.index else x.index.max()
    x_now = x.loc[[pred_date]]
    if x_now.isna().any(axis=1).iloc[0]:
        raise ValueError(f"Feature row for {pred_date} contains NaN values.")

    x_now_sc = scaler.transform(x_now)
    pred_label = int(model.predict(x_now_sc)[0])
    pred_proba_up = float(model.predict_proba(x_now_sc)[0, 1])

    return {
        "prediction_date": pred_date,
        "pred_label": pred_label,
        "pred_proba_up": pred_proba_up,
        "open_price": float(frame.loc[pred_date, "Open"]),
        "latest_price": float(frame.loc[pred_date, "Close"]),
    }


def _download_live_ohlcv(ticker: str, start: str, end: str, interval: str | None = None, period: str | None = None) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()

    kwargs = {"auto_adjust": True, "progress": False, "threads": False}
    if interval is not None:
        kwargs["interval"] = interval
    if period is not None:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end

    for _ in range(3):
        try:
            data = yf.download(ticker, **kwargs)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.rename_axis("Date").sort_index()
            if not data.empty:
                return data
        except Exception:
            pass

        try:
            if period is not None:
                alt = yf.Ticker(ticker).history(period=period, interval=interval or "1d", auto_adjust=True)
            else:
                alt = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            alt = alt.rename_axis("Date").sort_index()
            if not alt.empty:
                return alt
        except Exception:
            pass

    return pd.DataFrame()


def load_market_data(asset_key: str, prefer_live: bool = True) -> tuple[pd.DataFrame, str]:
    settings = _asset_settings(asset_key)
    raw_path = os.path.join(config.DATA_DIR, f"{settings['code']}_index.csv")
    processed_path = os.path.join(config.DATA_DIR, f"{settings['code']}_processed.csv")

    data = pd.DataFrame()
    source = ""

    if prefer_live:
        end_date = (pd.Timestamp(datetime.now().date()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = _download_live_ohlcv(settings["ticker"], start=config.START_DATE, end=end_date)
        if not data.empty:
            source = "yfinance"

    if data.empty and os.path.exists(raw_path):
        data = pd.read_csv(raw_path, parse_dates=["Date"]).set_index("Date").sort_index()
        source = "local raw csv"

    if data.empty and os.path.exists(processed_path):
        data = pd.read_csv(processed_path, parse_dates=["Date"]).set_index("Date").sort_index()
        source = "local processed csv"

    if data.empty and not prefer_live:
        end_date = (pd.Timestamp(datetime.now().date()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = _download_live_ohlcv(settings["ticker"], start=config.START_DATE, end=end_date)
        if not data.empty:
            source = "yfinance"

    if data.empty:
        raise FileNotFoundError(f"No usable data source for {settings['name']}. Tried yfinance, {raw_path}, and {processed_path}.")

    data = data[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "Close"])
    return data, source


def load_intraday_data(asset_key: str) -> pd.DataFrame:
    settings = _asset_settings(asset_key)
    intraday = _download_live_ohlcv(
        settings["ticker"],
        start=config.START_DATE,
        end=(pd.Timestamp(datetime.now().date()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="60m",
        period="730d",
    )
    if intraday.empty:
        raise ValueError("Intraday data is unavailable from yfinance for this asset right now.")

    return intraday[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "Close"])


def predict_daily(raw_daily: pd.DataFrame, today: pd.Timestamp) -> dict:
    windows = (config.MA_SHORT, config.MA_LONG, config.VOL_WINDOW, config.RSI_WINDOW, config.BB_WINDOW)
    out = _predict_from_frame(raw_daily, today, today, min_rows=200, windows=windows)
    out["input_date"] = out["prediction_date"] - pd.Timedelta(days=1)
    out["in_progress"] = out["prediction_date"].date() == today.date()
    out["frame_used"] = raw_daily
    return out


def predict_weekly(raw_daily: pd.DataFrame, today: pd.Timestamp) -> dict:
    weekly = raw_daily.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Open", "Close"])
    week_end = today + pd.Timedelta(days=(4 - today.weekday()) % 7)
    out = _predict_from_frame(weekly, week_end, week_end, min_rows=60, windows=scaled_windows(divisor=5))
    out["input_date"] = out["prediction_date"] - pd.Timedelta(days=7)
    out["in_progress"] = out["prediction_date"] >= today
    out["frame_used"] = weekly
    return out


def predict_monthly(raw_daily: pd.DataFrame, today: pd.Timestamp) -> dict:
    monthly = raw_daily.resample("ME").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Open", "Close"])
    month_end = today.to_period("M").to_timestamp("M")
    out = _predict_from_frame(monthly, month_end, month_end, min_rows=24, windows=scaled_windows(divisor=21))
    out["input_date"] = (out["prediction_date"].to_period("M") - 1).to_timestamp("M")
    out["in_progress"] = out["prediction_date"] >= month_end
    out["frame_used"] = monthly
    return out


def predict_quarterly(raw_daily: pd.DataFrame, today: pd.Timestamp) -> dict:
    quarterly = raw_daily.resample("QE").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Open", "Close"])
    quarter_end = today.to_period("Q").to_timestamp("Q")
    out = _predict_from_frame(quarterly, quarter_end, quarter_end, min_rows=12, windows=scaled_windows(divisor=63))
    out["input_date"] = (out["prediction_date"].to_period("Q") - 1).to_timestamp("Q")
    out["in_progress"] = out["prediction_date"] >= quarter_end
    out["frame_used"] = quarterly
    return out


def predict_yearly(raw_daily: pd.DataFrame, today: pd.Timestamp) -> dict:
    yearly = raw_daily.resample("YE").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Open", "Close"])
    year_end = today.to_period("Y").to_timestamp("Y")
    out = _predict_from_frame(yearly, year_end, year_end, min_rows=10, windows=scaled_windows(divisor=252))
    out["input_date"] = (out["prediction_date"].to_period("Y") - 1).to_timestamp("Y")
    out["in_progress"] = out["prediction_date"] >= year_end
    out["frame_used"] = yearly
    return out


def predict_1h(asset_key: str) -> dict:
    hourly = load_intraday_data(asset_key)
    now_ts = pd.Timestamp(hourly.index.max())
    target = now_ts.floor("h")
    out = _predict_from_frame(hourly, target, target, min_rows=500, windows=intraday_windows(hours_per_bar=1))
    out["input_date"] = out["prediction_date"] - pd.Timedelta(hours=1)
    out["in_progress"] = out["prediction_date"] >= now_ts.floor("h")
    out["frame_used"] = hourly
    return out


def predict_4h(asset_key: str) -> dict:
    hourly = load_intraday_data(asset_key)
    four_h = hourly.resample("4H").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Open", "Close"])
    now_ts = pd.Timestamp(four_h.index.max())
    target = now_ts.floor("4H")
    out = _predict_from_frame(four_h, target, target, min_rows=250, windows=intraday_windows(hours_per_bar=4))
    out["input_date"] = out["prediction_date"] - pd.Timedelta(hours=4)
    out["in_progress"] = out["prediction_date"] >= now_ts.floor("4H")
    out["frame_used"] = four_h
    return out


def horizon_predictor(horizon: str, asset_key: str, raw_daily: pd.DataFrame, today: pd.Timestamp) -> dict:
    if horizon == "1H":
        return predict_1h(asset_key)
    if horizon == "4H":
        return predict_4h(asset_key)
    if horizon == "Daily":
        return predict_daily(raw_daily, today)
    if horizon == "Weekly":
        return predict_weekly(raw_daily, today)
    if horizon == "Monthly":
        return predict_monthly(raw_daily, today)
    if horizon == "Quarterly":
        return predict_quarterly(raw_daily, today)
    if horizon == "Yearly":
        return predict_yearly(raw_daily, today)
    raise ValueError(f"Unsupported horizon: {horizon}")


def predict_all_horizons(asset_key: str, raw_daily: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for horizon in PERIOD_OPTIONS:
        try:
            pred = horizon_predictor(horizon, asset_key, raw_daily, today)
            rows.append({
                "Horizon": horizon,
                "Direction": "UP" if pred["pred_label"] == 1 else "DOWN/FLAT",
                "P(UP)": pred["pred_proba_up"],
                "Prediction Date": str(pd.Timestamp(pred["prediction_date"])),
                "Input Date": str(pd.Timestamp(pred["input_date"])),
                "Open": pred["open_price"],
                "Latest": pred["latest_price"],
                "In Progress": bool(pred.get("in_progress", False)),
            })
        except Exception as exc:
            rows.append({"Horizon": horizon, "Direction": "N/A", "P(UP)": np.nan, "Error": str(exc)})
    return pd.DataFrame(rows)


def make_price_chart(df: pd.DataFrame, asset_name: str) -> go.Figure:
    t = _theme_palette()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", line=dict(color=t["accent"], width=2.3), name="Close"))
    fig.update_layout(title=f"{asset_name} Price", template=t["plotly"], height=410, margin=dict(l=20, r=20, t=45, b=20))
    return fig


def make_probability_gauge(probability: float, asset_name: str, horizon: str) -> go.Figure:
    t = _theme_palette()
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": f"P(UP) | {asset_name} | {horizon}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": t["accent2"] if probability >= 0.5 else "#f59e0b"},
                "steps": [
                    {"range": [0, 50], "color": "rgba(255,120,120,0.12)"},
                    {"range": [50, 100], "color": "rgba(16,185,129,0.16)"},
                ],
            },
        )
    )
    fig.update_layout(template=t["plotly"], height=260, margin=dict(l=20, r=20, t=50, b=20))
    return fig


_inject_css()

if "nav" not in st.session_state:
    st.session_state.nav = "Forecast"

st.markdown('<div class="topbar">', unsafe_allow_html=True)
col_l, col_m, col_r = st.columns([2, 2, 1])
with col_l:
    st.markdown("### Market Direction Lab")
with col_m:
    st.session_state.nav = st.radio(
        "Navigation",
        options=["Forecast", "Comparison", "Data"],
        horizontal=True,
        label_visibility="collapsed",
        index=["Forecast", "Comparison", "Data"].index(st.session_state.nav),
    )
with col_r:
    label = "Switch to Light" if _theme_mode() == "dark" else "Switch to Dark"
    if st.button(label, use_container_width=True):
        _toggle_theme()
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>Minimal Forecast Console</h1>
        <p>Multi-asset, multi-horizon direction forecasts with live data fallback.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    selected_asset = st.selectbox("Asset", list(ASSET_OPTIONS.keys()), index=0)
    selected_horizon = st.selectbox("Horizon", PERIOD_OPTIONS, index=2)
    prefer_live_data = st.toggle("Prefer live data (yfinance)", value=True)
    lookback_points = st.slider("Chart points", min_value=120, max_value=3000, value=900, step=30)
    if st.button("Refresh data", use_container_width=True):
        st.rerun()
    st.caption("1H/4H use intraday yfinance data.")

asset_key = ASSET_OPTIONS[selected_asset]
settings = _asset_settings(asset_key)
clock_today = pd.Timestamp(datetime.now().date())

try:
    raw_daily, data_source = load_market_data(asset_key, prefer_live=prefer_live_data)
except Exception as exc:
    st.error(f"Unable to load data for {selected_asset}: {exc}")
    st.stop()

last_market_date = pd.Timestamp(raw_daily.index.max()).normalize()
ref_date = min(clock_today, last_market_date)

try:
    prediction = horizon_predictor(selected_horizon, asset_key, raw_daily, ref_date)
except Exception as exc:
    st.error(f"Prediction failed for {selected_asset} / {selected_horizon}: {exc}")
    st.stop()

comparison_df = predict_all_horizons(asset_key, raw_daily, ref_date)
frame_used = prediction.get("frame_used", raw_daily)
chart_df = frame_used.tail(lookback_points) if len(frame_used) > lookback_points else frame_used

direction_label = "UP" if prediction["pred_label"] == 1 else "DOWN/FLAT"

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><strong>Asset</strong><br>{settings["name"]}</div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><strong>Horizon</strong><br>{selected_horizon}</div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><strong>Direction</strong><br>{direction_label}</div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><strong>Prediction Time</strong><br>{pd.Timestamp(prediction["prediction_date"])}</div>', unsafe_allow_html=True)

st.caption(f"Data source: {data_source} | Latest daily date: {last_market_date.date()}")

if st.session_state.nav == "Forecast":
    left, right = st.columns((1.6, 1))
    with left:
        st.plotly_chart(make_price_chart(chart_df, settings["name"]), use_container_width=True)
    with right:
        st.plotly_chart(make_probability_gauge(prediction["pred_proba_up"], settings["name"], selected_horizon), use_container_width=True)
        st.markdown(
            f"""
            <div class=\"metric-card\">
                <strong>Open</strong>: {prediction['open_price']:.2f}<br>
                <strong>Latest close/last</strong>: {prediction['latest_price']:.2f}<br>
                <strong>Model input time</strong>: {pd.Timestamp(prediction['input_date'])}
            </div>
            """,
            unsafe_allow_html=True,
        )

    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Predicted direction", direction_label, f"P(UP) {prediction['pred_proba_up']:.1%}")
    with s2:
        st.metric("Prediction timestamp", str(pd.Timestamp(prediction["prediction_date"])))
    with s3:
        st.metric("Input timestamp", str(pd.Timestamp(prediction["input_date"])))

    if prediction.get("in_progress"):
        st.info(f"Current {PERIOD_LABELS[selected_horizon]} is still in progress, values can update intraperiod.")

elif st.session_state.nav == "Comparison":
    st.subheader(f"All Horizons | {settings['name']}")
    ok_rows = comparison_df[comparison_df.get("P(UP)", pd.Series(dtype=float)).notna()].copy()

    if not ok_rows.empty:
        bar = go.Figure(
            go.Bar(
                x=ok_rows["Horizon"],
                y=ok_rows["P(UP)"] * 100,
                marker_color=["#10b981" if p >= 0.5 else "#f59e0b" for p in ok_rows["P(UP)"]],
                text=[f"{p:.1%}" for p in ok_rows["P(UP)"]],
                textposition="outside",
            )
        )
        bar.update_layout(template=_theme_palette()["plotly"], height=360, yaxis_title="P(UP) %", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(bar, use_container_width=True)

    show_df = comparison_df.copy()
    if "P(UP)" in show_df.columns:
        show_df["P(UP)"] = show_df["P(UP)"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    for col in ["Open", "Latest"]:
        if col in show_df.columns:
            show_df[col] = show_df[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
    st.dataframe(show_df, use_container_width=True, hide_index=True)

else:
    st.subheader(f"Raw Data | {settings['name']}")
    st.dataframe(raw_daily.tail(150), use_container_width=True)
    st.download_button(
        label="Download latest daily data CSV",
        data=raw_daily.to_csv().encode("utf-8"),
        file_name=f"{settings['code']}_daily_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
