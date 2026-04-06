"""
ERCOT Market Intelligence Dashboard
Real-time grid monitoring, situational awareness, 1h/24h forecasts (GNN + XGBoost),
anomaly alerts with history, weather-vs-load correlation, outage tracking,
active alert banner, grid stress heatmap, and model comparison.
"""

import os
import logging
import requests
import pandas as pd
import numpy as np
import psycopg2
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(
    page_title="ERCOT Market Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

_log = logging.getLogger(__name__)

DB_URL = (os.environ.get("DATABASE_URL") or "").strip()
MODEL_SERVER = os.environ.get("MODEL_SERVER_URL", "http://model-server:8000")
REFRESH_SECS = int(os.environ.get("DASHBOARD_REFRESH_SECS", "60"))

if not DB_URL:
    st.error("DATABASE_URL is not set. Configure it for the dashboard container (see .env.example).")
    st.stop()


@st.cache_data(ttl=60)
def query_db(sql: str, params: tuple = None) -> pd.DataFrame:
    try:
        conn = psycopg2.connect(DB_URL)
        df = pd.read_sql(sql, conn, params=params)
        conn.close()
        return df
    except Exception:
        _log.exception("Dashboard SQL query failed")
        return pd.DataFrame()


def api_get(path: str) -> dict | None:
    try:
        r = requests.get(f"{MODEL_SERVER}{path}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ──────────────── Sidebar ────────────────

st.sidebar.title("ERCOT Intelligence")
st.sidebar.markdown("Real-time grid monitoring & ML forecasts")
days_back = st.sidebar.slider("Days to display", 1, 30, 7)
settlement_point = st.sidebar.selectbox(
    "Settlement Point",
    ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_BUSAVG"],
)
weather_zone = st.sidebar.selectbox(
    "Weather Zone",
    ["All Zones (Avg)", "houston", "dallas", "san_antonio", "west_texas", "corpus"],
)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)

if auto_refresh:
    @st.fragment(run_every=timedelta(seconds=REFRESH_SECS))
    def _auto_refresh_tick():
        st.caption(f"Auto-refreshing every {REFRESH_SECS}s — {datetime.utcnow().strftime('%H:%M:%S UTC')}")
    _auto_refresh_tick()

# ──────────────── Fetch API Data ────────────────

health = api_get("/health")
grid_status = api_get("/grid/status")
load_1h = api_get("/forecast/load?horizon=1")
load_24h = api_get("/forecast/load?horizon=24")
price_1h = api_get(f"/forecast/price?horizon=1&settlement_point={settlement_point}")
price_24h = api_get(f"/forecast/price?horizon=24&settlement_point={settlement_point}")
anomaly_data = api_get("/anomalies")
xgb_forecast = api_get("/forecast/load/xgboost")
active_alerts = api_get("/alerts/active")
load_compare = api_get("/forecast/load/compare")

# ──────────────── ALERT BANNER ────────────────

if active_alerts and active_alerts.get("count", 0) > 0:
    for alert in active_alerts["alerts"][:3]:
        sev = alert.get("severity", "info")
        icon = {"critical": "🔴", "warning": "🟡"}.get(sev, "🔵")
        st.error(f"{icon} **{alert.get('title', 'Alert')}** — {alert.get('description', '')[:200]}")

# ──────────────── Title + KPIs ────────────────

st.title("ERCOT Market Intelligence Dashboard")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

with c1:
    if grid_status and grid_status.get("load"):
        cur = grid_status["load"].get("current_mw")
        st.metric("Live Load", f"{cur:,.0f} MW" if cur else "N/A")
    else:
        st.metric("Live Load", "N/A")
with c2:
    val = f"{load_1h['forecast_mw']:,.0f} MW" if load_1h else "N/A"
    st.metric("GNN 1h", val)
with c3:
    val = f"{load_24h['forecast_mw']:,.0f} MW" if load_24h else "N/A"
    st.metric("GNN 24h", val)
with c4:
    val = f"${price_1h['forecast_usd_mwh']:.2f}" if price_1h else "N/A"
    st.metric("Price 1h", val)
with c5:
    if anomaly_data:
        count = anomaly_data["anomaly_count"]
        st.metric("Anomalies", count,
                  delta="ALERT" if count > 0 else "normal",
                  delta_color="inverse" if count > 0 else "normal")
    else:
        st.metric("Anomalies", "N/A")
with c6:
    n_alerts = active_alerts.get("count", 0) if active_alerts else 0
    st.metric("Active Alerts", n_alerts,
              delta="ACTION" if n_alerts > 0 else None,
              delta_color="inverse" if n_alerts > 0 else "normal")
with c7:
    n_models = len(health.get("models", [])) if health else 0
    st.metric("Models", n_models)

st.divider()

# ──────────────── Tabs ────────────────

tab_overview, tab_load, tab_price, tab_anomaly, tab_weather, tab_outage, tab_metrics = st.tabs([
    "Grid Overview", "Load Forecast", "Prices", "Anomalies", "Weather", "Outages & Alerts", "Model Metrics"
])

# ════════════════ TAB: Grid Overview ════════════════

with tab_overview:
    st.subheader("Real-Time Grid Situational Awareness")

    col_ov1, col_ov2 = st.columns([2, 1])

    with col_ov1:
        df_load = query_db(
            "SELECT ts, load_mw, forecast_mw, grid_stress FROM grid_load WHERE ts > NOW() - (%s * INTERVAL '1 day') ORDER BY ts",
            (int(days_back),),
        )
        if not df_load.empty:
            fig_ov = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                   row_heights=[0.7, 0.3],
                                   subplot_titles=("System Load (MW)", "Grid Stress (Actual - Forecast)"))
            fig_ov.add_trace(go.Scatter(x=df_load["ts"], y=df_load["load_mw"],
                                        name="Actual", line=dict(color="#1f77b4", width=1.5)), row=1, col=1)
            if df_load["forecast_mw"].notna().any():
                fig_ov.add_trace(go.Scatter(x=df_load["ts"], y=df_load["forecast_mw"],
                                            name="Day-Ahead Forecast", line=dict(color="#ff7f0e", dash="dash")), row=1, col=1)
            if df_load["grid_stress"].notna().any():
                colors = ["#d62728" if v > 0 else "#2ca02c" for v in df_load["grid_stress"].fillna(0)]
                fig_ov.add_trace(go.Bar(x=df_load["ts"], y=df_load["grid_stress"],
                                        name="Stress", marker_color=colors, opacity=0.7), row=2, col=1)
            fig_ov.update_layout(height=500, margin=dict(t=40, b=20), showlegend=True,
                                 legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig_ov, use_container_width=True)

    with col_ov2:
        st.markdown("**Grid Status Summary**")

        if grid_status and grid_status.get("load"):
            ld = grid_status["load"]
            st.metric("Current Load", f"{ld.get('current_mw', 0):,.0f} MW")
            if ld.get("forecast_mw"):
                st.metric("Day-Ahead Forecast", f"{ld['forecast_mw']:,.0f} MW")
            if ld.get("stress"):
                stress_val = ld["stress"]
                st.metric("Grid Stress", f"{stress_val:+,.0f} MW",
                          delta="Over-forecast" if stress_val > 0 else "Under-forecast",
                          delta_color="inverse" if abs(stress_val) > 2000 else "normal")

        st.markdown("---")
        st.markdown("**Model Forecasts**")
        compare_rows = []
        if load_1h:
            compare_rows.append({"Model": "GNN (1h)", "MW": f"{load_1h['forecast_mw']:,.0f}"})
        if load_24h:
            compare_rows.append({"Model": "GNN (24h)", "MW": f"{load_24h['forecast_mw']:,.0f}"})
        if xgb_forecast:
            compare_rows.append({"Model": "XGBoost", "MW": f"{xgb_forecast['forecast_mw']:,.0f}"})
        if compare_rows:
            st.dataframe(pd.DataFrame(compare_rows), hide_index=True, use_container_width=True)

    st.subheader("Load Heatmap — Hour of Day vs Day")
    if not df_load.empty:
        df_hm = df_load.copy()
        df_hm["ts"] = pd.to_datetime(df_hm["ts"])
        df_hm["hour"] = df_hm["ts"].dt.hour
        df_hm["date"] = df_hm["ts"].dt.date
        hm_pivot = df_hm.pivot_table(index="hour", columns="date", values="load_mw", aggfunc="mean")
        fig_hm = px.imshow(hm_pivot, color_continuous_scale="YlOrRd",
                           labels=dict(x="Date", y="Hour of Day", color="MW"),
                           aspect="auto")
        fig_hm.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_hm, use_container_width=True)

# ════════════════ TAB: Load Forecast ════════════════

with tab_load:
    st.subheader("Grid Load — Actual vs Forecast")
    df_load = query_db(
        "SELECT ts, load_mw, forecast_mw FROM grid_load WHERE ts > NOW() - (%s * INTERVAL '1 day') ORDER BY ts",
        (int(days_back),),
    )
    if not df_load.empty:
        fig_load = go.Figure()
        fig_load.add_trace(go.Scatter(x=df_load["ts"], y=df_load["load_mw"],
                                      name="Actual Load", line=dict(color="#1f77b4", width=1.5)))
        if df_load["forecast_mw"].notna().any():
            fig_load.add_trace(go.Scatter(x=df_load["ts"], y=df_load["forecast_mw"],
                                          name="EIA Day-Ahead", line=dict(color="#ff7f0e", dash="dash")))
        if load_1h:
            fig_load.add_hline(y=load_1h["forecast_mw"], line_dash="dot", line_color="green",
                               annotation_text=f"GNN 1h: {load_1h['forecast_mw']:,.0f} MW")
        if load_24h:
            fig_load.add_hline(y=load_24h["forecast_mw"], line_dash="dot", line_color="darkgreen",
                               annotation_text=f"GNN 24h: {load_24h['forecast_mw']:,.0f} MW",
                               annotation_position="bottom right")
        if xgb_forecast:
            fig_load.add_hline(y=xgb_forecast["forecast_mw"], line_dash="dashdot", line_color="orange",
                               annotation_text=f"XGB: {xgb_forecast['forecast_mw']:,.0f} MW",
                               annotation_position="top left")
        fig_load.update_layout(xaxis_title="Time", yaxis_title="MW", height=450,
                               margin=dict(t=30, b=30), legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig_load, use_container_width=True)
    else:
        st.info("No load data yet.")

    if load_compare and load_compare.get("models"):
        st.subheader("Load forecast — model comparison (live API)")
        rows = []
        for name, payload in load_compare["models"].items():
            if not payload:
                rows.append({"Model": name, "Forecast MW": "—", "Horizon": "—"})
                continue
            rows.append({
                "Model": name,
                "Forecast MW": f"{payload.get('forecast_mw', 0):,.0f}" if payload.get("forecast_mw") is not None else "—",
                "Horizon": f"{payload.get('horizon_hours', '')}h" if payload.get("horizon_hours") is not None else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"Source: `{MODEL_SERVER}/forecast/load/compare` at {load_compare.get('timestamp', '')}")

# ════════════════ TAB: Prices ════════════════

with tab_price:
    st.subheader(f"Settlement Point Prices — {settlement_point}")
    df_dam = query_db(
        "SELECT ts, price_usd_mwh FROM spp_prices WHERE ts > NOW() - (%s * INTERVAL '1 day') AND settlement_point = %s AND market_type = 'DAM' ORDER BY ts",
        (int(days_back), settlement_point),
    )
    df_rtm = query_db(
        "SELECT ts, price_usd_mwh FROM spp_prices WHERE ts > NOW() - (%s * INTERVAL '1 day') AND settlement_point = %s AND market_type = 'RTM' ORDER BY ts",
        (int(days_back), settlement_point),
    )

    fig_price = go.Figure()
    if not df_dam.empty:
        fig_price.add_trace(go.Scatter(x=df_dam["ts"], y=df_dam["price_usd_mwh"],
                                       name="DAM", line=dict(color="#2ca02c")))
    if not df_rtm.empty:
        fig_price.add_trace(go.Scatter(x=df_rtm["ts"], y=df_rtm["price_usd_mwh"],
                                       name="RTM", line=dict(color="#d62728")))
    if price_1h:
        fig_price.add_hline(y=price_1h["forecast_usd_mwh"], line_dash="dot", line_color="purple",
                            annotation_text=f"GNN 1h: ${price_1h['forecast_usd_mwh']:.2f}")
    if price_24h:
        fig_price.add_hline(y=price_24h["forecast_usd_mwh"], line_dash="dot", line_color="indigo",
                            annotation_text=f"GNN 24h: ${price_24h['forecast_usd_mwh']:.2f}",
                            annotation_position="bottom right")
    fig_price.update_layout(xaxis_title="Time", yaxis_title="$/MWh", height=450,
                            margin=dict(t=30, b=30), legend=dict(orientation="h", y=1.02))
    if not df_dam.empty or not df_rtm.empty:
        st.plotly_chart(fig_price, use_container_width=True)

        if not df_dam.empty and len(df_dam) > 24:
            st.subheader("Price Volatility")
            df_vol = df_dam.copy()
            df_vol["ts"] = pd.to_datetime(df_vol["ts"])
            df_vol = df_vol.set_index("ts")
            df_vol["rolling_std"] = df_vol["price_usd_mwh"].rolling(24).std()
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=df_vol.index, y=df_vol["rolling_std"],
                                         name="24h Rolling Std Dev", fill="tozeroy",
                                         line=dict(color="#9467bd")))
            fig_vol.update_layout(yaxis_title="$/MWh (Std Dev)", height=250, margin=dict(t=20, b=20))
            st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.info("No price data yet.")

# ════════════════ TAB: Anomalies ════════════════

with tab_anomaly:
    st.subheader("Anomaly Detection & Alerts")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("**Live Anomaly Detection (GNN Autoencoder)**")
        if anomaly_data and anomaly_data.get("anomalies"):
            df_anom = pd.DataFrame(anomaly_data["anomalies"])
            df_anom["timestamp"] = pd.to_datetime(df_anom["timestamp"])
            fig_anom = go.Figure()
            fig_anom.add_trace(go.Scatter(
                x=df_anom["timestamp"], y=df_anom["score"], mode="markers",
                marker=dict(size=10, color=df_anom["score"], colorscale="Reds",
                            showscale=True, colorbar=dict(title="Score")),
                text=df_anom["severity"], name="Anomalies",
            ))
            fig_anom.add_hline(y=anomaly_data["threshold"], line_dash="dash", line_color="red",
                               annotation_text=f"Threshold: {anomaly_data['threshold']:.4f}")
            fig_anom.update_layout(xaxis_title="Time", yaxis_title="Reconstruction Error",
                                   height=350, margin=dict(t=30, b=30))
            st.plotly_chart(fig_anom, use_container_width=True)
            with st.expander("Anomaly Details"):
                st.dataframe(df_anom, use_container_width=True)
        else:
            st.info("No live anomalies detected — grid behavior is within normal patterns.")

    with col_a2:
        st.markdown("**Anomaly History (Persisted)**")
        history = api_get(f"/anomalies/history?days={days_back}")
        if history and history.get("events"):
            df_hist = pd.DataFrame(history["events"])
            df_hist["detected_at"] = pd.to_datetime(df_hist["detected_at"])
            df_hist["severity"] = pd.to_numeric(df_hist["severity"], errors="coerce")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(
                x=df_hist["detected_at"], y=df_hist["severity"],
                marker=dict(color=df_hist["severity"], colorscale="YlOrRd"), name="Severity"))
            fig_hist.update_layout(xaxis_title="Time", yaxis_title="Severity Score",
                                   height=350, margin=dict(t=30, b=30))
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption(f"{history['count']} events in last {days_back} days")
        else:
            st.info("No historical anomaly events.")

# ════════════════ TAB: Weather ════════════════

with tab_weather:
    st.subheader("Weather Data & Load Correlation")

    zone_param = None if weather_zone == "All Zones (Avg)" else weather_zone
    if zone_param:
        df_weather = query_db(
            "SELECT ts, temperature_2m, wind_speed_10m, direct_radiation, COALESCE(relative_humidity, 0) as relative_humidity FROM weather WHERE ts > NOW() - (%s * INTERVAL '1 day') AND zone = %s ORDER BY ts",
            (int(days_back), zone_param),
        )
    else:
        df_weather = query_db(
            "SELECT ts, AVG(temperature_2m) as temperature_2m, AVG(wind_speed_10m) as wind_speed_10m, AVG(direct_radiation) as direct_radiation FROM weather WHERE ts > NOW() - (%s * INTERVAL '1 day') GROUP BY ts ORDER BY ts",
            (int(days_back),),
        )

    if not df_weather.empty:
        fig_wx = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                               subplot_titles=("Temperature (°C)", "Wind Speed (m/s)", "Solar Radiation (W/m²)"))
        fig_wx.add_trace(go.Scatter(x=df_weather["ts"], y=df_weather["temperature_2m"],
                                    line=dict(color="#e377c2"), name="Temp"), row=1, col=1)
        fig_wx.add_trace(go.Scatter(x=df_weather["ts"], y=df_weather["wind_speed_10m"],
                                    line=dict(color="#17becf"), name="Wind"), row=2, col=1)
        fig_wx.add_trace(go.Scatter(x=df_weather["ts"], y=df_weather["direct_radiation"],
                                    line=dict(color="#bcbd22"), name="Solar"), row=3, col=1)
        fig_wx.update_layout(height=500, showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig_wx, use_container_width=True)

        st.subheader("Multi-Zone Weather Comparison")
        df_zones = query_db(
            "SELECT ts, zone, temperature_2m FROM weather WHERE ts > NOW() - (%s * INTERVAL '1 day') ORDER BY ts",
            (int(days_back),),
        )
        if not df_zones.empty and df_zones["zone"].nunique() > 1:
            fig_zones = px.line(df_zones, x="ts", y="temperature_2m", color="zone",
                                title="Temperature Across ERCOT Zones")
            fig_zones.update_layout(height=350, margin=dict(t=40, b=20))
            st.plotly_chart(fig_zones, use_container_width=True)

        st.subheader("Weather vs Load — Scatter Correlation")
        df_corr = query_db(
            """SELECT g.ts, g.load_mw, w.temperature_2m, w.wind_speed_10m, w.direct_radiation
               FROM grid_load g
               JOIN weather w ON date_trunc('hour', g.ts) = date_trunc('hour', w.ts)
               WHERE g.ts > NOW() - (%s * INTERVAL '1 day')
                 AND (w.zone = 'houston' OR w.zone IS NULL)
               ORDER BY g.ts""",
            (int(days_back),),
        )
        if not df_corr.empty and len(df_corr) > 10:
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                fig_sc = px.scatter(df_corr, x="temperature_2m", y="load_mw", opacity=0.5, trendline="ols",
                                    labels={"temperature_2m": "Temp (°C)", "load_mw": "Load (MW)"},
                                    title="Temperature vs Load")
                fig_sc.update_layout(height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig_sc, use_container_width=True)
            with col_s2:
                fig_sc2 = px.scatter(df_corr, x="wind_speed_10m", y="load_mw", opacity=0.5, trendline="ols",
                                     labels={"wind_speed_10m": "Wind (m/s)", "load_mw": "Load (MW)"},
                                     title="Wind vs Load")
                fig_sc2.update_layout(height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig_sc2, use_container_width=True)
            with col_s3:
                fig_sc3 = px.scatter(df_corr, x="direct_radiation", y="load_mw", opacity=0.5, trendline="ols",
                                     labels={"direct_radiation": "Solar (W/m²)", "load_mw": "Load (MW)"},
                                     title="Solar vs Load")
                fig_sc3.update_layout(height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig_sc3, use_container_width=True)

            with st.expander("Correlation Matrix"):
                numeric_cols = [c for c in ["load_mw", "temperature_2m", "wind_speed_10m", "direct_radiation"] if c in df_corr.columns]
                corr_matrix = df_corr[numeric_cols].corr()
                fig_cm = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                                   zmin=-1, zmax=1, title="Pearson Correlation")
                fig_cm.update_layout(height=350, margin=dict(t=40, b=20))
                st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("Load + Temperature Overlay")
        df_overlay = query_db(
            """SELECT g.ts, g.load_mw, w.temperature_2m
               FROM grid_load g
               JOIN weather w ON date_trunc('hour', g.ts) = date_trunc('hour', w.ts)
               WHERE g.ts > NOW() - (%s * INTERVAL '1 day') AND (w.zone = 'houston' OR w.zone IS NULL)
               ORDER BY g.ts""",
            (int(days_back),),
        )
        if not df_overlay.empty:
            fig_ov = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ov.add_trace(go.Scatter(x=df_overlay["ts"], y=df_overlay["load_mw"],
                                        name="Load (MW)", line=dict(color="#1f77b4")), secondary_y=False)
            fig_ov.add_trace(go.Scatter(x=df_overlay["ts"], y=df_overlay["temperature_2m"],
                                        name="Temp (°C)", line=dict(color="#d62728", dash="dot")), secondary_y=True)
            fig_ov.update_yaxes(title_text="Load (MW)", secondary_y=False)
            fig_ov.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
            fig_ov.update_layout(height=350, margin=dict(t=20, b=20), legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig_ov, use_container_width=True)
    else:
        st.info("No weather data yet.")

# ════════════════ TAB: Outages & Alerts ════════════════

with tab_outage:
    st.subheader("Grid Outages & System Alerts")

    col_out1, col_out2 = st.columns(2)

    with col_out1:
        st.markdown("**System Condition Snapshots**")
        outages = api_get(f"/outages?days={days_back}")
        if outages and outages.get("outages"):
            df_out = pd.DataFrame(outages["outages"])
            stress_count = len(df_out[df_out["status"] == "high_stress"]) if "status" in df_out.columns else 0
            normal_count = len(df_out) - stress_count

            col_met1, col_met2 = st.columns(2)
            col_met1.metric("Total Snapshots", len(df_out))
            col_met2.metric("High Stress Events", stress_count,
                           delta="ALERT" if stress_count > 0 else None,
                           delta_color="inverse" if stress_count > 0 else "normal")

            if "capacity_mw" in df_out.columns and "ts" in df_out.columns:
                df_out["ts"] = pd.to_datetime(df_out["ts"])
                fig_cap = go.Figure()
                fig_cap.add_trace(go.Scatter(x=df_out["ts"], y=df_out["capacity_mw"],
                                             name="Reserve Capacity (MW)", fill="tozeroy",
                                             line=dict(color="#2ca02c")))
                fig_cap.update_layout(yaxis_title="Reserve MW", height=300, margin=dict(t=20, b=20))
                st.plotly_chart(fig_cap, use_container_width=True)

            with st.expander("Raw Outage Data"):
                st.dataframe(df_out, use_container_width=True)
        else:
            st.info("No outage data yet — system conditions monitor may still be starting.")

    with col_out2:
        st.markdown("**Alert History**")
        alerts = api_get(f"/alerts?days={days_back}")
        if alerts and alerts.get("alerts"):
            df_alerts = pd.DataFrame(alerts["alerts"])
            df_alerts["ts"] = pd.to_datetime(df_alerts["ts"])

            for sev in ["critical", "warning", "info"]:
                sev_count = len(df_alerts[df_alerts["severity"] == sev])
                if sev_count > 0:
                    icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(sev, "")
                    st.caption(f"{icon} {sev.upper()}: {sev_count} alerts")

            fig_alerts = go.Figure()
            sev_map = {"critical": 3, "warning": 2, "info": 1}
            df_alerts["sev_num"] = df_alerts["severity"].map(sev_map).fillna(0)
            fig_alerts.add_trace(go.Scatter(
                x=df_alerts["ts"], y=df_alerts["sev_num"], mode="markers",
                marker=dict(size=12,
                            color=df_alerts["sev_num"],
                            colorscale=[[0, "blue"], [0.5, "yellow"], [1, "red"]],
                            showscale=False),
                text=df_alerts["title"], name="Alerts"))
            fig_alerts.update_layout(yaxis=dict(tickvals=[1, 2, 3], ticktext=["Info", "Warning", "Critical"]),
                                     height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig_alerts, use_container_width=True)

            with st.expander("Alert Details"):
                cols = [c for c in ["ts", "alert_id", "severity", "title", "description", "source", "acknowledged"] if c in df_alerts.columns]
                st.dataframe(df_alerts[cols] if cols else df_alerts, use_container_width=True)
            with st.expander("Acknowledge an alert"):
                ack_id = st.text_input("Alert ID (UUID from the table above)", key="ack_alert_id")
                if st.button("Mark acknowledged", key="ack_btn"):
                    if not (ack_id or "").strip():
                        st.warning("Enter an alert_id.")
                    else:
                        try:
                            r = requests.post(
                                f"{MODEL_SERVER}/alerts/{ack_id.strip()}/acknowledge", timeout=10
                            )
                            if r.status_code == 200:
                                st.success("Acknowledged.")
                            else:
                                st.warning(f"Server returned {r.status_code}: {r.text[:200]}")
                        except Exception as ex:
                            st.error(str(ex))
        else:
            st.info("No alerts recorded yet.")

# ════════════════ TAB: Model Metrics ════════════════

with tab_metrics:
    st.subheader("Model Performance Tracking")
    df_metrics = query_db(
        "SELECT ts, model_name, metric_name, metric_value, horizon FROM model_metrics ORDER BY ts DESC LIMIT 200"
    )
    if not df_metrics.empty:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            mape_df = df_metrics[df_metrics["metric_name"] == "mape"]
            if not mape_df.empty:
                fig_mape = go.Figure()
                for model in sorted(mape_df["model_name"].unique()):
                    m_df = mape_df[mape_df["model_name"] == model].sort_values("ts")
                    fig_mape.add_trace(go.Scatter(x=m_df["ts"], y=m_df["metric_value"],
                                                   name=model, mode="lines+markers"))
                fig_mape.update_layout(title="MAPE Over Time", yaxis_title="MAPE (%)", height=350)
                st.plotly_chart(fig_mape, use_container_width=True)

        with col_m2:
            mae_df = df_metrics[df_metrics["metric_name"] == "mae"]
            if not mae_df.empty:
                fig_mae = go.Figure()
                for model in sorted(mae_df["model_name"].unique()):
                    m_df = mae_df[mae_df["model_name"] == model].sort_values("ts")
                    fig_mae.add_trace(go.Scatter(x=m_df["ts"], y=m_df["metric_value"],
                                                  name=model, mode="lines+markers"))
                fig_mae.update_layout(title="MAE Over Time", yaxis_title="MAE", height=350)
                st.plotly_chart(fig_mae, use_container_width=True)

        with st.expander("Latest Metrics — All Models"):
            latest = df_metrics.sort_values("ts", ascending=False).drop_duplicates(
                subset=["model_name", "metric_name"], keep="first"
            )[["model_name", "metric_name", "metric_value", "horizon", "ts"]].reset_index(drop=True)
            st.dataframe(latest, use_container_width=True)
    else:
        st.info("No model metrics yet.")

# ──────────────── Footer ────────────────

st.divider()
st.caption(f"Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | "
           f"Model Server: {MODEL_SERVER} | Compose stack (ingest, trainer, API, DB, dashboard)")
