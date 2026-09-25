"""
dashboard.py -- Interactive Streamlit app for the TTC Transit Service
Reliability Analysis. Run from the repo root with:

    streamlit run app/dashboard.py

Reads the star-schema CSVs in dashboard/powerbi_data/ (exported by
`python ttc_pipeline.py --powerbi-export`) and reuses the pipeline's own
feature engineering and station matcher, so the numbers here come from
the same code as the offline analysis.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ttc_pipeline import (  # noqa: E402
    DELAY_THRESHOLD_MIN,
    engineer_features,
    normalize_station_name,
)

DATA_DIR = os.path.join(ROOT, "dashboard", "powerbi_data")

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TTC Subway Delays: Reliability Analysis",
    page_icon="🚇",
    layout="wide",
)

INK = "#1b2024"
TTC_RED = "#c8102e"
LINE_COLORS = {"Line 1": "#e8ae00", "Line 2": "#00923f", "Line 4": "#a21a68"}
BAR = "#3e4a54"
NEUTRAL = "#9aa6af"
GOOD = "#2f7a3d"
SEVERITY_SCALE = [[0, "#dce1e4"], [0.5, TTC_RED], [1, "#5e0b16"]]
PLOTLY_TEMPLATE = "simple_white"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #f7f8f7; }}
    section[data-testid="stSidebar"] {{
        background-color: #eef0ee;
        border-right: 1px solid #dde1de;
    }}
    div[data-testid="stMetric"] {{
        background-color: white;
        border: 1px solid #e3e6e4;
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ font-size: 0.82rem; color: #5c656d; }}
    div[data-testid="stMetricLabel"] > div {{
        white-space: normal !important; overflow: visible !important; text-overflow: unset !important;
    }}
    div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
    h1, h2, h3 {{ color: {INK}; }}
    .hero-card {{
        background: {INK};
        color: white;
        padding: 26px 32px 22px 32px;
        border-radius: 14px;
        margin-bottom: 8px;
        position: relative;
        overflow: hidden;
    }}
    .hero-card .stripe {{ display: flex; height: 7px; position: absolute; top: 0; left: 0; right: 0; }}
    .hero-card .stripe span {{ flex: 1; }}
    .hero-card h1 {{ color: white; margin: 6px 0 6px 0; font-size: 1.7rem; }}
    .hero-card p {{ color: #c9d0d5; margin-bottom: 0; font-size: 0.95rem; }}
    .callout-box {{
        background-color: #fff8e1;
        border-left: 4px solid {LINE_COLORS["Line 1"]};
        padding: 14px 18px; border-radius: 6px; margin: 12px 0;
    }}
    .callout-warn {{
        background-color: #fdecee;
        border-left: 4px solid {TTC_RED};
        padding: 14px 18px; border-radius: 6px; margin: 12px 0;
    }}
    .callout-good {{
        background-color: #e8f5ec;
        border-left: 4px solid {LINE_COLORS["Line 2"]};
        padding: 14px 18px; border-radius: 6px; margin: 12px 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero-card">
        <div class="stripe"><span style="background:{LINE_COLORS['Line 1']}"></span>
        <span style="background:{LINE_COLORS['Line 2']}"></span>
        <span style="background:{LINE_COLORS['Line 4']}"></span></div>
        <h1>🚇 Where and when does Toronto's subway stall?</h1>
        <p>23,701 real TTC subway delay events (Jan 2024 to Jun 2026) from City of Toronto Open Data,
        joined with Environment Canada weather. End-to-end: messy-data cleaning, a SQL star schema,
        exploratory analysis, and a Random Forest vs Gradient Boosting model trained live in this app.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
LINE_CODE_TO_NAME = {"YU": "Line 1", "BD": "Line 2", "SHP": "Line 4"}
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]


def pretty_station(raw_name: str) -> str:
    name = raw_name.replace(" STATION", "").title()
    return "Cedarvale (Eglinton West)" if name == "Eglinton West" else name


@st.cache_data
def load_data():
    f = pd.read_csv(os.path.join(DATA_DIR, "fact_delay_events.csv"), parse_dates=["date_id"])
    stn = pd.read_csv(os.path.join(DATA_DIR, "dim_station.csv"))
    codes = pd.read_csv(os.path.join(DATA_DIR, "dim_delay_code.csv"))
    weather = pd.read_csv(os.path.join(DATA_DIR, "fact_weather_daily.csv"), parse_dates=["date_id"])
    fi = pd.read_csv(os.path.join(DATA_DIR, "feature_importances.csv"))

    df = f.merge(stn[["station_id", "station_name"]], on="station_id", how="left")
    df = df.merge(codes[["code", "code_description"]], on="code", how="left")
    df = df.merge(weather.drop(columns=["total_snow_cm"]), on="date_id", how="left")

    # A handful of raw rows carry bus/streetcar route names or "YUS" in the
    # line field; fall back to the station's own line for those.
    station_line = dict(zip(stn["station_name"], stn["line"]))
    df["line"] = df["line"].replace({"YUS": "YU"})
    bad = ~df["line"].isin(LINE_CODE_TO_NAME)
    df.loc[bad, "line"] = df.loc[bad, "station_name"].map(station_line)
    df["line_name"] = df["line"].map(LINE_CODE_TO_NAME)

    df["station"] = df["station_name"].map(pretty_station)
    df["cause"] = df["code_description"].fillna(df["code"])
    df["day_of_week"] = df["date_id"].dt.day_name()
    df["is_weekend"] = df["date_id"].dt.dayofweek >= 5
    month = df["date_id"].dt.month
    df["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["Winter", "Spring", "Summer"], default="Fall",
    )
    df["month"] = df["date_id"].dt.to_period("M").dt.to_timestamp()
    df["delay_prone"] = df["min_delay"] >= DELAY_THRESHOLD_MIN
    return df, weather, fi


df, weather_df, saved_fi = load_data()
UNKNOWN = "Unknown / Other"

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Filters")
line_filter = st.sidebar.multiselect(
    "Subway line", options=list(LINE_CODE_TO_NAME.values()), default=list(LINE_CODE_TO_NAME.values())
)
min_d, max_d = df["date_id"].min().date(), df["date_id"].max().date()
date_range = st.sidebar.slider("Date range", min_value=min_d, max_value=max_d, value=(min_d, max_d), format="MMM YYYY")
season_filter = st.sidebar.multiselect("Season", options=SEASON_ORDER, default=SEASON_ORDER)
day_type = st.sidebar.radio("Days", ["All days", "Weekdays", "Weekends"], horizontal=True)
hour_range = st.sidebar.slider("Hour of day", 0, 23, (0, 23))
min_delay_len = st.sidebar.slider(
    "Minimum delay length (min)", 0, 60, 0,
    help="Raise this to focus on the more serious delays only.",
)

working = df[
    df["line_name"].isin(line_filter)
    & df["season"].isin(season_filter)
    & df["hour"].between(*hour_range)
    & (df["date_id"].dt.date >= date_range[0])
    & (df["date_id"].dt.date <= date_range[1])
].copy()
if day_type == "Weekdays":
    working = working[~working["is_weekend"]]
elif day_type == "Weekends":
    working = working[working["is_weekend"]]
if min_delay_len > 0:
    working = working[working["min_delay"] >= min_delay_len]

st.sidebar.markdown("---")
st.sidebar.metric("Delay events in view", f"{len(working):,}")
st.sidebar.metric("Hours of delay in view", f"{working['min_delay'].sum() / 60:,.0f}")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: [TTC Subway Delay Data](https://open.toronto.ca/dataset/ttc-subway-delay-data/), "
    "City of Toronto Open Data; daily weather from Environment Canada (Toronto City station). "
    "Code: [GitHub](https://github.com/HitakshiKathiriya/ttc-transit-reliability)."
)

EMPTY_MSG = "No delay events match the current filters. Widen the filters in the sidebar."


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def group_stats(d: pd.DataFrame, by) -> pd.DataFrame:
    g = d.groupby(by).agg(
        events=("min_delay", "size"),
        avg_delay=("min_delay", "mean"),
        median_delay=("min_delay", "median"),
        prone_rate=("delay_prone", "mean"),
        total_min=("min_delay", "sum"),
    )
    return g


def dual_axis_chart(x, counts, avgs, title, x_title, height=380):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=x, y=counts, name="Delay events", marker_color=BAR), secondary_y=False)
    fig.add_trace(
        go.Scatter(x=x, y=avgs, name="Avg delay (min)", mode="lines+markers",
                   line=dict(color=TTC_RED, width=3), marker=dict(size=6)),
        secondary_y=True,
    )
    fig.update_layout(title=title, template=PLOTLY_TEMPLATE, height=height,
                      margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.2),
                      hovermode="x unified")
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text="Delay events", secondary_y=False)
    fig.update_yaxes(title_text="Avg delay (min)", secondary_y=True, rangemode="tozero")
    return fig


def fmt_hour(h: int) -> str:
    return "12am" if h == 0 else f"{h}am" if h < 12 else "12pm" if h == 12 else f"{h - 12}pm"


# ---------------------------------------------------------------------------
# Schematic network map geometry (Lines 1, 2 and 4)
# ---------------------------------------------------------------------------
BD = ["Kipling", "Islington", "Royal York", "Old Mill", "Jane", "Runnymede", "High Park", "Keele",
      "Dundas West", "Lansdowne", "Dufferin", "Ossington", "Christie", "Bathurst", "Spadina",
      "St George", "Bay", "Bloor-Yonge", "Sherbourne", "Castle Frank", "Broadview", "Chester", "Pape",
      "Donlands", "Greenwood", "Coxwell", "Woodbine", "Main Street", "Victoria Park", "Warden", "Kennedy"]
Y2, XY, XU, XS = 470, 640, 510, 460


def _bd_x(i):
    if i <= 13:
        return 54 + i * 29
    return {14: 460, 15: 510, 16: 575, 17: 640}.get(i, 675 + (i - 18) * 34)


POS, LABEL_SIDE = {}, {}
for i, n in enumerate(BD):
    POS[n] = (_bd_x(i), Y2)
    LABEL_SIDE[n] = "below" if i <= 13 else "above"
for i, n in enumerate(["Rosedale", "Summerhill", "St Clair", "Davisville", "Eglinton", "Lawrence",
                       "York Mills", "Sheppard-Yonge", "North York Centre", "Finch"]):
    POS[n] = (XY, 380 - 30 * i)
    LABEL_SIDE[n] = "right"
for i, n in enumerate(["Wellesley", "College", "Dundas", "Queen", "King"]):
    POS[n] = (XY, Y2 + 45 + 35 * i)
    LABEL_SIDE[n] = "right"
for i, n in enumerate(["Museum", "Queens Park", "St Patrick", "Osgoode", "St Andrew"]):
    POS[n] = (XU, Y2 + 45 + 35 * i)
    LABEL_SIDE[n] = "right"
POS["Union"] = ((XY + XU) / 2, 688)
LABEL_SIDE["Union"] = "under"
for i, n in enumerate(["Dupont", "St Clair West", "Cedarvale (Eglinton West)", "Glencairn", "Lawrence West",
                       "Yorkdale", "Wilson", "Sheppard West", "Downsview Park", "Finch West"]):
    POS[n] = (XS, Y2 - 30 * (i + 1))
    LABEL_SIDE[n] = "right"
for i, n in enumerate(["York University", "Pioneer Village", "Highway 407", "Vaughan Metropolitan Centre"]):
    POS[n] = (XS - 29 * (i + 1), 170 - 28 * (i + 1))
    LABEL_SIDE[n] = "right"
for i, n in enumerate(["Bayview", "Bessarion", "Leslie", "Don Mills"]):
    POS[n] = (XY + 110 + 55 * i, POS["Sheppard-Yonge"][1])
    LABEL_SIDE[n] = "under"
LABEL_SIDE.update({"Bloor-Yonge": "right-below", "St George": "left-below", "Spadina": "left-above",
                   "Sheppard-Yonge": "right-above", "North York Centre": "left",
                   "York University": "left"})

L1_PATH = (
    [POS[n] for n in ["Vaughan Metropolitan Centre", "Highway 407", "Pioneer Village", "York University",
                      "Finch West", "Downsview Park", "Sheppard West", "Wilson", "Yorkdale", "Lawrence West",
                      "Glencairn", "Cedarvale (Eglinton West)", "St Clair West", "Dupont"]]
    + [(XS, Y2 + 6), (XU, Y2 + 6)]
    + [POS[n] for n in ["Museum", "Queens Park", "St Patrick", "Osgoode", "St Andrew"]]
    + [(XU + 12, 682), POS["Union"], (XY - 12, 682)]
    + [POS[n] for n in ["King", "Queen", "Dundas", "College", "Wellesley", "Bloor-Yonge", "Rosedale",
                        "Summerhill", "St Clair", "Davisville", "Eglinton", "Lawrence", "York Mills",
                        "Sheppard-Yonge", "North York Centre", "Finch"]]
)
L2_PATH = [POS[n] for n in BD]
L4_PATH = [POS[n] for n in ["Sheppard-Yonge", "Bayview", "Bessarion", "Leslie", "Don Mills"]]


def network_map(stats: pd.DataFrame, size_metric: str, selected: str | None):
    fig = go.Figure()
    for path, name in [(L1_PATH, "Line 1"), (L2_PATH, "Line 2"), (L4_PATH, "Line 4")]:
        dim = name not in line_filter
        fig.add_trace(go.Scatter(
            x=[p[0] for p in path], y=[p[1] for p in path], mode="lines", hoverinfo="skip",
            line=dict(color=LINE_COLORS[name], width=7), opacity=0.2 if dim else 1, name=name,
            showlegend=False,
        ))

    names = list(POS)
    s = stats.reindex(names)
    counts = s["events"].fillna(0)
    size_val = counts if size_metric == "Delay events" else s["total_min"].fillna(0)
    max_val = max(size_val.max(), 1)
    sizes = 7 + np.sqrt(size_val / max_val) * 22
    solid = s.loc[s["events"] >= 20, "avg_delay"]
    cmin = float(solid.quantile(0.05)) if len(solid) else 0.0
    cmax = float(solid.quantile(0.95)) if len(solid) else 10.0

    fig.add_trace(go.Scatter(
        x=[POS[n][0] for n in names], y=[POS[n][1] for n in names], mode="markers",
        marker=dict(
            size=sizes, color=s["avg_delay"].fillna(cmin), colorscale=SEVERITY_SCALE, cmin=cmin, cmax=max(cmax, cmin + 0.1),
            line=dict(color=[TTC_RED if n == selected else INK for n in names],
                      width=[4 if n == selected else 1 for n in names]),
            colorbar=dict(title="Avg delay<br>(min)", thickness=12, len=0.5, y=0.75),
            opacity=0.95,
        ),
        customdata=np.array(list(zip(names, counts, s["avg_delay"].fillna(0), s["prone_rate"].fillna(0) * 100)), dtype=object),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,} delays<br>"
                      "%{customdata[2]:.1f} min average<br>%{customdata[3]:.0f}% lasted 5+ min"
                      "<extra>Click to drill in</extra>",
        showlegend=False,
    ))

    offsets = {
        "right": dict(xanchor="left", yanchor="middle", xshift=1, textangle=0),
        "left": dict(xanchor="right", yanchor="middle", xshift=-1, textangle=0),
        "above": dict(xanchor="left", yanchor="bottom", yshift=1, textangle=-90),
        "below": dict(xanchor="right", yanchor="top", yshift=-1, textangle=-90),
        "under": dict(xanchor="center", yanchor="top", yshift=-1, textangle=0),
        "right-below": dict(xanchor="left", yanchor="top", xshift=0.7, yshift=-0.7, textangle=0),
        "left-below": dict(xanchor="right", yanchor="top", xshift=-0.7, yshift=-0.7, textangle=0),
        "left-above": dict(xanchor="right", yanchor="bottom", xshift=-0.7, yshift=0.7, textangle=0),
        "right-above": dict(xanchor="left", yanchor="bottom", xshift=0.7, yshift=0.7, textangle=0),
    }
    interchanges = {"Bloor-Yonge", "St George", "Spadina", "Sheppard-Yonge"}
    for n, sz in zip(names, sizes):
        o = dict(offsets[LABEL_SIDE[n]])
        gap = sz / 2 + 4
        o["xshift"] = o.get("xshift", 0) * gap
        o["yshift"] = o.get("yshift", 0) * gap
        label = "Cedarvale" if n.startswith("Cedarvale") else n
        bold = n in interchanges or n == selected
        fig.add_annotation(
            x=POS[n][0], y=POS[n][1], text=f"<b>{label}</b>" if bold else label, showarrow=False,
            font=dict(size=10.5, color=TTC_RED if n == selected else INK), **o,
        )
    for num, color, (x, y), fg in [("1", LINE_COLORS["Line 1"], (340, 24), INK),
                                   ("2", LINE_COLORS["Line 2"], (18, Y2), "white"),
                                   ("4", LINE_COLORS["Line 4"], (POS["Don Mills"][0] + 34, POS["Don Mills"][1]), "white")]:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[f"<b>{num}</b>"],
                                 marker=dict(size=24, color=color), textfont=dict(color=fg, size=13),
                                 hoverinfo="skip", showlegend=False))

    fig.update_xaxes(visible=False, range=[0, 1110], fixedrange=True)
    fig.update_yaxes(visible=False, range=[725, 0], fixedrange=True)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=760, margin=dict(t=10, b=10, l=10, r=10),
                      plot_bgcolor="white", clickmode="event+select", dragmode=False)
    return fig


# ---------------------------------------------------------------------------
# Modeling helpers (reuse the pipeline's own feature engineering)
# ---------------------------------------------------------------------------
BASE_FEATURES = ["hour", "day_of_week", "season", "station_name", "line", "bound", "mode", "is_weekend"]
FEATURE_GROUPS = {
    "Weather (temperature, precipitation)": ["mean_temp_c", "total_precip_mm"],
    "Rush-hour flag": ["is_rush_hour"],
    "Rolling station delay history (7 and 30 days, leakage-safe)": [
        "rolling_7d_station_delays", "rolling_30d_station_delays"],
}
NICE_FEATURE = {
    "hour": "Hour of day", "day_of_week": "Day of week", "season": "Season", "station_name": "Station",
    "line": "Line", "bound": "Direction", "mode": "Mode", "is_weekend": "Weekend",
    "mean_temp_c": "Mean temperature", "total_precip_mm": "Precipitation", "is_rush_hour": "Rush hour",
    "rolling_7d_station_delays": "Station delays, past 7 days",
    "rolling_30d_station_delays": "Station delays, past 30 days",
}


@st.cache_data(show_spinner=False)
def build_features():
    model_df = df[["date_id", "hour", "min_delay", "min_gap", "code", "mode", "bound", "line", "station_name",
                   "day_of_week", "is_weekend", "season", "mean_temp_c", "total_precip_mm"]].copy()
    X, y, encoders = engineer_features(model_df)
    return X, y, encoders


@st.cache_resource(show_spinner=False)
def train_models(feature_cols: tuple, run_cv: bool):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

    X, y, _ = build_features()
    X = X[list(feature_cols)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    out = {}
    for name, model in [("Gradient Boosting", gb), ("Random Forest", rf)]:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        out[name] = {"model": model, "auc": roc_auc_score(y_test, proba), "fpr": fpr, "tpr": tpr}
        if run_cv:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
            out[name]["cv_mean"], out[name]["cv_std"] = scores.mean(), scores.std()
    return out, y.mean()


# ---------------------------------------------------------------------------
# Tabbed navigation
# ---------------------------------------------------------------------------
(tab_overview, tab_map, tab_time, tab_cause, tab_weather,
 tab_quality, tab_model, tab_findings) = st.tabs(
    ["🏠 Overview", "🗺️ Stations", "⏰ Time Patterns", "🔍 Root Causes",
     "🌦️ Weather & Seasons", "🧹 Data Quality", "🤖 Model", "📝 Findings"]
)

# --- Overview tab ---
with tab_overview:
    st.subheader("What this project answers")
    st.markdown(
        """
        Which stations, times and conditions are linked to subway delays in Toronto, and can we predict
        which delays will last **5 minutes or more**? This dashboard runs on the real delay log published
        by the City of Toronto, cleaned and modeled with the same pipeline code that's in the repo.
        """
    )
    if len(working):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Delay events", f"{len(working):,}")
        c2.metric("Average delay", f"{working['min_delay'].mean():.1f} min")
        c3.metric("Median delay", f"{working['min_delay'].median():.0f} min")
        c4.metric("Lasted 5+ minutes", f"{working['delay_prone'].mean():.1%}",
                  help="The model's target: share of delay events at or above the 5-minute threshold.")
        c5.metric("Hours of delay", f"{working['min_delay'].sum() / 60:,.0f}")

        col1, col2 = st.columns([3, 2])
        with col1:
            by_line = group_stats(working, "line_name").reindex(line_filter).dropna()
            fig = go.Figure(go.Bar(
                x=by_line.index, y=by_line["events"],
                marker_color=[LINE_COLORS[n] for n in by_line.index],
                text=[f"{v:,.0f}" for v in by_line["events"]], textposition="outside",
                customdata=by_line["avg_delay"], hovertemplate="%{x}<br>%{y:,} delays<br>%{customdata:.1f} min avg<extra></extra>",
            ))
            fig.update_layout(title="Delay events by line", yaxis_title="Delay events",
                              template=PLOTLY_TEMPLATE, height=360, margin=dict(t=50, b=40))
            st.plotly_chart(fig, width="stretch")
        with col2:
            monthly = working.groupby("month").size()
            fig = go.Figure(go.Scatter(x=monthly.index, y=monthly.values, mode="lines",
                                       line=dict(color=INK, width=2), fill="tozeroy",
                                       fillcolor="rgba(62,74,84,0.12)"))
            fig.update_layout(title="Delay events per month", template=PLOTLY_TEMPLATE, height=360,
                              margin=dict(t=50, b=40), yaxis_title="Delay events")
            st.plotly_chart(fig, width="stretch")
    else:
        st.warning(EMPTY_MSG)

    st.markdown(
        """
        <div class="callout-box">
        <b>How to read this dashboard:</b> use the <b>sidebar filters</b> to slice every tab at once.
        Start at <b>Stations</b> and click any station on the map to drill in, then see <b>Time Patterns</b>
        for when delays cluster, <b>Root Causes</b> for why, and <b>Weather &amp; Seasons</b> for the role of
        cold and snow. <b>Data Quality</b> shows how 535 messy station labels became 71 clean ones, and
        <b>Model</b> retrains Random Forest and Gradient Boosting live so you can test the feature experiments
        yourself.
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Stations tab ---
with tab_map:
    st.subheader("Where delays happen")
    st.caption("Circle size shows volume, colour shows average delay length. Click a station to drill in below.")
    if len(working):
        station_stats = group_stats(working[working["station"] != UNKNOWN], "station")
        size_metric = st.radio("Size circles by", ["Delay events", "Total minutes lost"], horizontal=True)

        if "drill_station" not in st.session_state:
            st.session_state.drill_station = station_stats["events"].idxmax()
        event = st.plotly_chart(
            network_map(station_stats, size_metric, st.session_state.drill_station),
            width="stretch", on_select="rerun", selection_mode="points", key="network_map",
            config={"displayModeBar": False},
        )
        points = (event or {}).get("selection", {}).get("points", []) if event else []
        clicked = next((p["customdata"][0] for p in points if p.get("customdata")), None)
        if clicked and clicked != st.session_state.get("last_map_click"):
            st.session_state.last_map_click = clicked
            st.session_state.drill_station = clicked
            st.rerun()

        st.markdown("#### Station ranking")
        r1, r2 = st.columns([1, 1])
        rank_by = r1.selectbox("Rank stations by", ["Delay events", "Average delay length", "Share lasting 5+ min",
                                                    "Total minutes lost"])
        min_events = r2.slider("Minimum delay events (for averages)", 10, 300, 50, step=10)
        col_map = {"Delay events": "events", "Average delay length": "avg_delay",
                   "Share lasting 5+ min": "prone_rate", "Total minutes lost": "total_min"}
        ranked = station_stats[station_stats["events"] >= min_events].sort_values(col_map[rank_by], ascending=False)
        st.dataframe(
            ranked.head(15).reset_index().rename(columns={
                "station": "Station", "events": "Delay events", "avg_delay": "Avg delay (min)",
                "median_delay": "Median (min)", "prone_rate": "Lasted 5+ min", "total_min": "Total minutes"}),
            width="stretch", hide_index=True,
            column_config={
                "Avg delay (min)": st.column_config.NumberColumn(format="%.1f"),
                "Median (min)": st.column_config.NumberColumn(format="%.0f"),
                "Lasted 5+ min": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "Total minutes": st.column_config.NumberColumn(format="%d"),
            },
        )
        busiest = station_stats.sort_values("events", ascending=False).head(15)
        leaders = ", ".join(busiest.index[:3])
        worst_avg = busiest["avg_delay"].idxmax()
        st.markdown(
            f"""
            <div class="callout-warn">
            <b>Read raw counts with care:</b> {leaders} lead on volume in this view, but busy terminals and
            interchanges also see the most trains. Among the 15 busiest stations, <b>{worst_avg}</b> has the
            longest average delay ({busiest.loc[worst_avg, "avg_delay"]:.1f} min), a different kind of problem
            station. A true ranking would normalize by scheduled service, which isn't in this dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### Station drill-down")
        options = station_stats.sort_values("events", ascending=False).index.tolist()
        if st.session_state.drill_station not in options:
            st.session_state.drill_station = options[0]
        chosen = st.selectbox("Station", options, key="drill_station")
        sd = working[working["station"] == chosen]
        net_avg = working["min_delay"].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Delay events", f"{len(sd):,}")
        m2.metric("Average delay", f"{sd['min_delay'].mean():.1f} min",
                  delta=f"{sd['min_delay'].mean() - net_avg:+.1f} min vs network", delta_color="inverse")
        m3.metric("Lasted 5+ minutes", f"{sd['delay_prone'].mean():.0%}")
        top_cause = sd["cause"].value_counts()
        m4.metric("Most common cause", top_cause.index[0] if len(top_cause) else "—")

        d1, d2 = st.columns(2)
        with d1:
            hourly_share = sd.groupby("hour").size().reindex(range(24), fill_value=0) / max(len(sd), 1)
            net_share = working.groupby("hour").size().reindex(range(24), fill_value=0) / len(working)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[fmt_hour(h) for h in range(24)], y=hourly_share * 100, name=chosen,
                                 marker_color=TTC_RED))
            fig.add_trace(go.Scatter(x=[fmt_hour(h) for h in range(24)], y=net_share * 100, name="Whole network",
                                     mode="lines", line=dict(color=INK, dash="dot", width=2)))
            fig.update_layout(title="When this station's delays happen vs the network",
                              yaxis_title="Share of delays (%)", template=PLOTLY_TEMPLATE, height=360,
                              margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
        with d2:
            daily = sd.groupby(sd["date_id"].dt.normalize()).size()
            if len(daily):
                full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
                rolling = daily.reindex(full, fill_value=0).rolling(30).sum().dropna()
                fig = go.Figure(go.Scatter(x=rolling.index, y=rolling.values, mode="lines",
                                           line=dict(color=INK, width=2)))
                fig.update_layout(title="Delays in the past 30 days (rolling)", yaxis_title="Delay events",
                                  template=PLOTLY_TEMPLATE, height=360, margin=dict(t=50, b=40))
                st.plotly_chart(fig, width="stretch")
        st.caption("Tip: click a terminal like Kipling or Kennedy on the map, then compare its rolling trend with a quieter station.")
    else:
        st.warning(EMPTY_MSG)

# --- Time Patterns tab ---
with tab_time:
    st.subheader("When delays happen")
    if len(working):
        hs = group_stats(working, "hour").reindex(range(24))
        st.plotly_chart(
            dual_axis_chart([fmt_hour(h) for h in range(24)], hs["events"], hs["avg_delay"],
                            "Frequent but short at rush hour, rare but long overnight", "Hour of day"),
            width="stretch",
        )
        c1, c2 = st.columns([2, 3])
        with c1:
            ds = group_stats(working, "day_of_week").reindex(DOW_ORDER)
            st.plotly_chart(
                dual_axis_chart([d[:3] for d in DOW_ORDER], ds["events"], ds["avg_delay"],
                                "By day of week", "Day of week"),
                width="stretch",
            )
        with c2:
            heat_metric = st.radio("Heatmap shows", ["Delay events", "Average delay (min)", "Share lasting 5+ min"],
                                   horizontal=True)
            metric_col = {"Delay events": "events", "Average delay (min)": "avg_delay",
                          "Share lasting 5+ min": "prone_rate"}[heat_metric]
            grid = group_stats(working, ["day_of_week", "hour"])[metric_col].unstack().reindex(
                index=DOW_ORDER, columns=range(24))
            fig = go.Figure(go.Heatmap(
                z=grid.values, x=[fmt_hour(h) for h in range(24)], y=[d[:3] for d in DOW_ORDER],
                colorscale=SEVERITY_SCALE, hoverongaps=False,
                hovertemplate="%{y} %{x}<br>%{z:.2f}<extra></extra>" if metric_col != "events"
                else "%{y} %{x}<br>%{z:,} delays<extra></extra>",
                zmax=np.nanpercentile(grid.values, 97) if metric_col == "avg_delay" else None,
            ))
            fig.update_layout(title=f"{heat_metric} by day and hour", template=PLOTLY_TEMPLATE, height=380,
                              margin=dict(t=50, b=40), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")

        sunday = ds.loc["Sunday", "avg_delay"]
        weekday = ds.loc[DOW_ORDER[:5], "avg_delay"].mean()
        st.markdown(
            f"""
            <div class="callout-box">
            <b>Two different problems:</b> rush hours carry the most delays, but the longest ones come in the
            early morning. Sundays follow the same pattern: in this view they average <b>{sunday:.1f} min</b>
            per delay vs <b>{weekday:.1f} min</b> on weekdays. Reduced service means less slack to recover when
            something goes wrong.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(EMPTY_MSG)

# --- Root Causes tab ---
with tab_cause:
    st.subheader("What causes delays")
    if len(working):
        c1, c2, c3 = st.columns(3)
        cause_rank = c1.radio("Rank causes by", ["Frequency", "Average length", "Total minutes lost"],
                              horizontal=True)
        top_n = c2.slider("Number of causes", 5, 20, 10)
        min_occ = c3.slider("Minimum occurrences", 5, 200, 30, step=5,
                            help="Stops rare one-off causes from topping the average-length ranking.")
        cs = group_stats(working, "cause")
        cs = cs[cs["events"] >= min_occ]
        sort_col = {"Frequency": "events", "Average length": "avg_delay", "Total minutes lost": "total_min"}[cause_rank]
        top = cs.sort_values(sort_col, ascending=False).head(top_n).iloc[::-1]
        label = lambda s: s if len(s) <= 45 else s[:43] + "…"  # noqa: E731
        fig = go.Figure(go.Bar(
            y=[label(c) for c in top.index], x=top[sort_col], orientation="h",
            marker_color=BAR if sort_col == "events" else TTC_RED,
            customdata=np.array(list(zip(top.index, top["events"], top["avg_delay"])), dtype=object),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,} delays<br>"
                          "%{customdata[2]:.1f} min average<extra></extra>",
        ))
        fig.update_layout(title=f"Top {top_n} causes by {cause_rank.lower()}",
                          xaxis_title={"events": "Delay events", "avg_delay": "Avg delay (min)",
                                       "total_min": "Total minutes"}[sort_col],
                          template=PLOTLY_TEMPLATE, height=max(380, 34 * top_n), margin=dict(t=50, b=40, l=10))
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### When each top cause happens")
        top8 = working["cause"].value_counts().head(8).index
        mat = (working[working["cause"].isin(top8)].groupby(["cause", "hour"]).size()
               .unstack(fill_value=0).reindex(columns=range(24), fill_value=0))
        mat = mat.div(mat.sum(axis=1), axis=0) * 100
        fig = go.Figure(go.Heatmap(
            z=mat.values, x=[fmt_hour(h) for h in range(24)], y=[label(c) for c in mat.index],
            colorscale=SEVERITY_SCALE, hovertemplate="%{y}<br>%{x}: %{z:.1f}% of this cause<extra></extra>",
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(t=20, b=40, l=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Each row sums to 100%, so you can compare the daily rhythm of causes with very different volumes.")
        st.markdown(
            """
            <div class="callout-box">
            <b>People, not machines:</b> the three most frequent causes (disorderly patron, passenger
            assistance alarm, train door monitoring) are about passenger behaviour rather than mechanical
            failure. That points to station staffing and customer response, not just maintenance, as levers.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(EMPTY_MSG)

# --- Weather & Seasons tab ---
with tab_weather:
    st.subheader("Weather and seasons")
    if len(working):
        monthly = working.groupby("month").agg(events=("min_delay", "size"), avg_delay=("min_delay", "mean"))
        w = weather_df.copy()
        w["month"] = w["date_id"].dt.to_period("M").dt.to_timestamp()
        temp = w.groupby("month")["mean_temp_c"].mean().reindex(monthly.index)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=monthly.index, y=monthly["events"], name="Delay events",
                             marker_color=[TTC_RED if m.month in (12, 1, 2) else BAR for m in monthly.index]),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=temp.index, y=temp.values, name="Mean temperature (°C)", mode="lines+markers",
                                 line=dict(color=LINE_COLORS["Line 2"], width=3)), secondary_y=True)
        fig.update_layout(title="Monthly delays vs temperature (winter months in red)", template=PLOTLY_TEMPLATE,
                          height=400, margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.15),
                          hovermode="x unified")
        fig.update_yaxes(title_text="Delay events", secondary_y=False)
        fig.update_yaxes(title_text="Mean temp (°C)", secondary_y=True)
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            daily = working.groupby(working["date_id"].dt.normalize()).agg(
                events=("min_delay", "size"), temp=("mean_temp_c", "first"), season=("season", "first")).dropna()
            fig = go.Figure()
            season_colors = {"Winter": TTC_RED, "Spring": LINE_COLORS["Line 2"],
                             "Summer": LINE_COLORS["Line 1"], "Fall": "#d98c3a"}
            for sname in SEASON_ORDER:
                sub = daily[daily["season"] == sname]
                fig.add_trace(go.Scatter(x=sub["temp"], y=sub["events"], mode="markers", name=sname,
                                         marker=dict(color=season_colors[sname], size=6, opacity=0.55)))
            if len(daily) > 10:
                slope, intercept = np.polyfit(daily["temp"], daily["events"], 1)
                xs = np.linspace(daily["temp"].min(), daily["temp"].max(), 50)
                fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Linear trend",
                                         line=dict(color=INK, width=2, dash="dash")))
                corr = daily["temp"].corr(daily["events"])
            fig.update_layout(title="Delays per day vs daily temperature", xaxis_title="Mean temperature (°C)",
                              yaxis_title="Delays that day", template=PLOTLY_TEMPLATE, height=400,
                              margin=dict(t=50, b=40), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
            if len(daily) > 10:
                st.caption(f"Correlation between daily temperature and delay count in this view: r = {corr:.2f}.")
        with c2:
            ss = group_stats(working, "season").reindex(SEASON_ORDER).dropna()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ss.index, y=ss["avg_delay"], marker_color=[season_colors[s] for s in ss.index],
                                 text=[f"{v:.1f} min" for v in ss["avg_delay"]], textposition="outside",
                                 customdata=ss["events"],
                                 hovertemplate="%{x}<br>%{y:.1f} min average<br>%{customdata:,} delays<extra></extra>"))
            fig.update_layout(title="Average delay length by season", yaxis_title="Avg delay (min)",
                              template=PLOTLY_TEMPLATE, height=400, margin=dict(t=50, b=40))
            st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
            <div class="callout-box">
            <b>What the model learned:</b> once real temperature was added as a feature, the importance of
            <i>season</i> dropped from about 0.09 to 0.01. Weather didn't add new signal so much as replace
            season with a more precise measure of the same thing.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(EMPTY_MSG)

# --- Data Quality tab ---
with tab_quality:
    st.subheader("Data quality: fixing messy station names")
    st.caption("TTC's open dataset cuts the station field off at 22 characters, and mixes in segments, "
               "directions and typos. Nothing downstream is trustworthy until this is fixed.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Distinct raw station values", "535")
    c2.metric("Clean stations after matching", "71")
    c3.metric("Records matched", "99.6%")

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = go.Figure(go.Bar(
            x=["First pass", "Typos, renames,<br>interchanges", "Full real dataset"], y=[92.3, 97.2, 99.6],
            marker_color=[NEUTRAL, BAR, LINE_COLORS["Line 2"]], text=["92.3%", "97.2%", "99.6%"],
            textposition="outside",
        ))
        fig.update_layout(title="Match rate as the matcher improved", yaxis_title="Records matched (%)",
                          yaxis_range=[85, 101], template=PLOTLY_TEMPLATE, height=380, margin=dict(t=50, b=40))
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.markdown("#### Try the matcher")
        st.caption("This runs `normalize_station_name()` from the pipeline live.")
        examples = ["SPADINA TO WILSON STAT", "APPROACHING KIPLING", "563 SHERBOURNE (SHERBO",
                    "YONGE BD STATION", "CEDARVALE STATION", "WISLON STATION", "MAIN STATION",
                    "CHESTER STATION - DEPA", "GUNN BUILDING", "SHEPPARD LINE"]
        example = st.selectbox("Pick a real-style raw value", examples)
        raw = st.text_input("…or type your own", value=example)
        result = normalize_station_name(raw)
        if result == "UNKNOWN / OTHER":
            st.markdown(
                f'<div class="callout-warn"><code>{raw}</code> → <b>left unmatched</b><br>'
                "Infrastructure sites and line-level references are deliberately not forced onto a station.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<div class="callout-good"><code>{raw}</code> → <b>{result.title()}</b></div>',
                        unsafe_allow_html=True)

    st.markdown(
        """
        <div class="callout-box">
        <b>How the matcher works:</b> instead of assuming the station name comes first, it searches for every
        real station as a whole word anywhere in the text and picks the one that appears earliest (longest
        name wins ties). That keeps the origin station for segments like "Spadina to Wilson", handles prefixes
        like "Approaching Kipling", and a small alias table covers typos, renamed stations (Cedarvale) and the
        Bloor-Yonge and Sheppard-Yonge interchanges.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Leakage check on the rolling features")
    st.markdown(
        "Each event's rolling 7- and 30-day station delay count is shifted forward one day before summing, "
        "so an event's own day never feeds its own feature. Verified directly: the first date in the data "
        "shows 0 for both windows, because there is no earlier history to leak from."
    )

# --- Model tab ---
with tab_model:
    st.subheader("Can we predict which delays last 5+ minutes?")
    st.caption("Binary classifier: given a delay event, will it last at least 5 minutes? Both models use identical "
               "features and the same 80/20 stratified split. Trained live on all 23,701 events (sidebar filters "
               "don't apply here).")

    chosen_groups = st.multiselect(
        "Features on top of the baseline (hour, day, season, station, line, direction, weekend)",
        list(FEATURE_GROUPS), default=list(FEATURE_GROUPS),
        help="Recreates the project's feature experiments. Remove groups to see how little AUC moves.",
    )
    run_cv = st.checkbox("Also run 5-fold cross-validation (slower, about a minute the first time)", value=False)
    feature_cols = tuple(BASE_FEATURES + [c for g in chosen_groups for c in FEATURE_GROUPS[g]])

    with st.spinner("Training Random Forest and Gradient Boosting..."):
        results, base_rate = train_models(feature_cols, run_cv)

    m1, m2, m3 = st.columns(3)
    gb_r, rf_r = results["Gradient Boosting"], results["Random Forest"]
    m1.metric("Gradient Boosting ROC-AUC (holdout)", f"{gb_r['auc']:.3f}",
              help=f"5-fold CV: {gb_r['cv_mean']:.3f} ± {gb_r['cv_std']:.3f}" if run_cv else None)
    m2.metric("Random Forest ROC-AUC (holdout)", f"{rf_r['auc']:.3f}",
              help=f"5-fold CV: {rf_r['cv_mean']:.3f} ± {rf_r['cv_std']:.3f}" if run_cv else None)
    m3.metric("Base rate (delays lasting 5+ min)", f"{base_rate:.1%}")
    if run_cv:
        st.caption(f"5-fold cross-validated ROC-AUC: Gradient Boosting {gb_r['cv_mean']:.3f} ± {gb_r['cv_std']:.3f}, "
                   f"Random Forest {rf_r['cv_mean']:.3f} ± {rf_r['cv_std']:.3f}.")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for name, color in [("Gradient Boosting", TTC_RED), ("Random Forest", NEUTRAL)]:
            r = results[name]
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"{name} ({r['auc']:.3f})",
                                     line=dict(color=color, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Coin flip (0.500)",
                                 line=dict(color=INK, dash="dash", width=1)))
        fig.update_layout(title="ROC curves on the holdout set", xaxis_title="False positive rate",
                          yaxis_title="True positive rate", template=PLOTLY_TEMPLATE, height=420,
                          margin=dict(t=50, b=40), legend=dict(x=0.45, y=0.08))
        st.plotly_chart(fig, width="stretch")
    with col2:
        imp = pd.Series(gb_r["model"].feature_importances_, index=feature_cols).sort_values()
        imp = imp[imp > 0]
        fig = go.Figure(go.Bar(
            y=[NICE_FEATURE.get(c, c) for c in imp.index], x=imp.values * 100, orientation="h",
            marker_color=[TTC_RED if v >= imp.nlargest(3).min() else NEUTRAL for v in imp.values],
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(title="What the Gradient Boosting model relies on", xaxis_title="Share of importance (%)",
                          template=PLOTLY_TEMPLATE, height=420, margin=dict(t=50, b=40, l=10))
        st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
        <div class="callout-warn">
        <b>The real finding is a ceiling, not a model problem.</b> Across five experiments (baseline, model swap,
        + weather, + rush-hour flag, + rolling history) ROC-AUC stayed between 0.62 and 0.65, and the two models
        land within about 0.01 of each other. Try removing feature groups above and watch how little changes.
        Beating this likely needs a different kind of data, such as live vehicle positions or ridership counts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### Delay risk predictor")
    st.caption("If a delay happens under these conditions, how likely is it to last 5+ minutes? "
               "Uses the Gradient Boosting model trained above.")
    X_all, _, encoders = build_features()
    p1, p2, p3 = st.columns(3)
    station_raw = sorted(encoders["station_name"].classes_)
    station_raw = [s for s in station_raw if s != "UNKNOWN / OTHER"]
    pick_station = p1.selectbox("Station", station_raw, format_func=pretty_station,
                                index=station_raw.index("KIPLING STATION") if "KIPLING STATION" in station_raw else 0)
    pick_hour = p2.slider("Hour of day", 0, 23, 8, format="%d:00")
    pick_day = p3.selectbox("Day of week", DOW_ORDER)
    p4, p5, p6 = st.columns(3)
    pick_month = p4.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                                        "Nov", "Dec"])
    pick_temp = p5.slider("Mean temperature (°C)", -25, 32, -5)
    pick_precip = p6.slider("Precipitation (mm)", 0.0, 40.0, 0.0, step=0.5)

    month_idx = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(pick_month) + 1
    pick_season = "Winter" if month_idx in (12, 1, 2) else "Spring" if month_idx <= 5 else "Summer" if month_idx <= 8 else "Fall"
    st_rows = df[df["station_name"] == pick_station]
    st_line = st_rows["line"].mode().iat[0] if len(st_rows) else "YU"
    st_bound = st_rows["bound"].dropna().mode().iat[0] if st_rows["bound"].notna().any() else "N"
    st_idx = X_all.index[X_all["station_name"] == encoders["station_name"].transform([pick_station])[0]]
    typical_7 = X_all.loc[st_idx, "rolling_7d_station_delays"].median() if len(st_idx) else 0
    typical_30 = X_all.loc[st_idx, "rolling_30d_station_delays"].median() if len(st_idx) else 0

    def enc(col, val):
        classes = list(encoders[col].classes_)
        return classes.index(str(val)) if str(val) in classes else 0

    row = {
        "hour": enc("hour", pick_hour), "day_of_week": enc("day_of_week", pick_day),
        "season": enc("season", pick_season), "station_name": enc("station_name", pick_station),
        "line": enc("line", st_line), "bound": enc("bound", st_bound), "mode": enc("mode", "SUBWAY"),
        "is_weekend": int(pick_day in ("Saturday", "Sunday")),
        "is_rush_hour": int(pick_hour in (7, 8, 9, 16, 17, 18)),
        "mean_temp_c": pick_temp, "total_precip_mm": pick_precip,
        "rolling_7d_station_delays": typical_7, "rolling_30d_station_delays": typical_30,
    }
    prob = gb_r["model"].predict_proba(pd.DataFrame([row])[list(feature_cols)])[:, 1][0]
    r1, r2 = st.columns([1, 2])
    r1.metric("Chance the delay lasts 5+ min", f"{prob:.0%}", delta=f"{(prob - base_rate) * 100:+.0f} pts vs average", delta_color="inverse")
    fig = go.Figure(go.Indicator(
        mode="gauge", value=prob * 100,
        gauge=dict(axis=dict(range=[0, 100], ticksuffix="%"), bar=dict(color=TTC_RED),
                   threshold=dict(line=dict(color=INK, width=3), value=base_rate * 100),
                   steps=[dict(range=[0, 100], color="#eef0ee")]),
    ))
    fig.update_layout(height=180, margin=dict(t=10, b=10, l=30, r=30))
    r2.plotly_chart(fig, width="stretch")
    st.caption("The black marker is the overall base rate. Station delay history is set to that station's typical "
               "7- and 30-day counts. With AUC around 0.64, treat this as a tilt in the odds, not a forecast.")

# --- Findings tab ---
with tab_findings:
    st.subheader("Findings and recommendations")
    st.markdown(
        """
        **1. Volume and severity are different problems.** Delays pile up at rush hour (8am, 4 to 5pm), but the
        longest ones happen overnight and early morning (4 to 5am). Sundays have the fewest delays but the
        longest average. Where service is thin, one incident does more damage.

        **2. Most delays are about people, not equipment.** The top three causes (disorderly patron, passenger
        assistance alarm, door monitoring) are passenger-related. Staffing and response time at busy stations
        are as much a reliability lever as maintenance.

        **3. Winter delays last longest.** Winter averages 8.7 minutes per delay vs 7.4 to 7.8 in the other
        seasons, even though spring logs slightly more events. Cold weather makes each incident harder to clear.

        **4. Raw station counts mislead.** Bloor-Yonge, Kipling and Kennedy lead on count largely because they
        are interchanges and terminals with heavy traffic. Ranking the busiest stations by average length
        (Eglinton, Warden, St Clair) is arguably the more useful "problem station" signal.

        **5. Prediction hits a feature ceiling.** Schedule, location and weather features plateau at about 0.64
        ROC-AUC regardless of model. Station and hour carry most of the signal.

        ---
        **Limitations**
        - Station counts aren't normalized by scheduled train volume, which isn't in this dataset.
        - About 0.4% of records are maintenance sites or line-level references, deliberately left unmatched.
        - The delay log only records incidents, so the model predicts severity given a delay, not whether one occurs.

        **Next steps**
        - Add bus and streetcar delay data (same Open Data structure), which would make the mode feature meaningful.
        - Normalize by scheduled service, and test live vehicle position data to push past the AUC ceiling.
        """
    )

st.markdown("---")
st.caption(
    "Built by Hitakshi Kathiriya — Data Analyst / Data Scientist. "
    "Data: TTC Subway Delay Data (City of Toronto Open Data, Jan 2024 to Jun 2026) and Environment Canada "
    "daily weather. Code: github.com/HitakshiKathiriya/ttc-transit-reliability"
)
