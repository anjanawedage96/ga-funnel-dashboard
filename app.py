#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import date

st.set_page_config(page_title="GA Funnel & ROAS Dashboard", layout="wide")
st.title("Google Analytics Funnel & ROAS Dashboard")

# --- Data loader ---
@st.cache_data
def load_data(path_or_buffer):
    df = pd.read_csv(path_or_buffer, parse_dates=["date"])
    # sanity checks
    expected = {"project","date","sessions","revenue","ad_spend","bounce_rate",
                "impressions","clicks","transactions"}
    missing = expected - set(df.columns)
    if missing:
        st.warning(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    return df

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload GA CSV", type=["csv"], help="Use the dataset we generated earlier.")
use_sample = st.sidebar.checkbox("Use sample file name (ga_year_demo_3_projects.csv)", value=True)

df = None
if uploaded is not None:
    df = load_data(uploaded)
elif use_sample:
    try:
        df = load_data("ga_year_demo_3_projects.csv")
    except Exception:
        st.info("Upload your CSV to begin (the sample file wasn't found).")
        st.stop()
else:
    st.info("Upload your CSV to begin.")
    st.stop()

# --- Filters ---
min_d, max_d = df["date"].min().date(), df["date"].max().date()
st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Date range",
    (min_d, max_d),
    min_value=min_d,
    max_value=max_d
)
if isinstance(date_range, tuple):
    start_d, end_d = date_range
else:
    start_d = date_range
    end_d = date_range

projects = sorted(df["project"].dropna().unique().tolist())
selected_projects = st.sidebar.multiselect("Project(s)", projects, default=projects)

# Apply filters
mask = (
    (df["date"].dt.date >= start_d) &
    (df["date"].dt.date <= end_d) &
    (df["project"].isin(selected_projects))
)
fdf = df.loc[mask].copy()
if fdf.empty:
    st.warning("No data in the selected filter range.")
    st.stop()

# --- KPI calculations ---
# Weighted bounce rate = sum(sessions*bounce_rate) / sum(sessions)
fdf["bounces_est"] = fdf["sessions"] * fdf["bounce_rate"]  # bounce_rate stored as 0–1 in the dataset
total_sessions = int(fdf["sessions"].sum())
total_revenue = float(fdf["revenue"].sum())
total_spend   = float(fdf["ad_spend"].sum())
total_bounces = float(fdf["bounces_est"].sum())

roas = (total_revenue / total_spend) if total_spend > 0 else np.nan
weighted_bounce = (total_bounces / total_sessions) if total_sessions > 0 else np.nan

# --- KPI tiles ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sessions", f"{total_sessions:,}")
c2.metric("Revenue (€)", f"{total_revenue:,.2f}")
c3.metric("ROAS", "-" if np.isnan(roas) else f"{roas:.2f}")
c4.metric("Bounce Rate", "-" if np.isnan(weighted_bounce) else f"{weighted_bounce:.2%}")

st.markdown("---")

# --- Helper dataframes ---
fdf["month"] = fdf["date"].dt.to_period("M").dt.to_timestamp()
monthly = (fdf.groupby(["month", "project"])
             .agg(impressions=("impressions","sum"),
                  clicks=("clicks","sum"),
                  sessions=("sessions","sum"),
                  transactions=("transactions","sum"),
                  revenue=("revenue","sum"),
                  ad_spend=("ad_spend","sum"))
             .reset_index())
monthly["ctr"] = np.where(monthly["impressions"]>0, monthly["clicks"]/monthly["impressions"], np.nan)
monthly["conversion_rate"] = np.where(monthly["sessions"]>0, monthly["transactions"]/monthly["sessions"], np.nan)
monthly["roas"] = np.where(monthly["ad_spend"]>0, monthly["revenue"]/monthly["ad_spend"], np.nan)

# --- Charts ---
# 1) Traffic trends (Sessions over time by project)
st.subheader("Traffic Trends")
line_sessions = alt.Chart(monthly).mark_line(point=True).encode(
    x=alt.X("month:T", title="Month"),
    y=alt.Y("sessions:Q", title="Sessions"),
    color=alt.Color("project:N", legend=alt.Legend(title="Project")),
    tooltip=["month:T","project:N","sessions:Q"]
).properties(height=300)
st.altair_chart(line_sessions, use_container_width=True)

# 2) Funnel (Impressions → Clicks → Sessions → Transactions) for current filters
st.subheader("Funnel (filtered)")
agg = {
    "Impressions": int(fdf["impressions"].sum()),
    "Clicks": int(fdf["clicks"].sum()),
    "Sessions": int(fdf["sessions"].sum()),
    "Transactions": int(fdf["transactions"].sum())
}
funnel_df = pd.DataFrame({"Stage": list(agg.keys()), "Value": list(agg.values())})
funnel_chart = alt.Chart(funnel_df).mark_bar().encode(
    x=alt.X("Stage:N", sort=None),
    y=alt.Y("Value:Q"),
    tooltip=["Stage","Value"]
).properties(height=280)
st.altair_chart(funnel_chart, use_container_width=True)

# 3) ROAS by project (bar, filtered date range)
st.subheader("ROAS by Project")
roas_proj = (fdf.groupby("project", as_index=False)
               .agg(revenue=("revenue","sum"), ad_spend=("ad_spend","sum")))
roas_proj["roas"] = np.where(roas_proj["ad_spend"]>0, roas_proj["revenue"]/roas_proj["ad_spend"], np.nan)
roas_chart = alt.Chart(roas_proj).mark_bar().encode(
    x=alt.X("project:N", title="Project", sort="-y"),
    y=alt.Y("roas:Q", title="ROAS"),
    tooltip=["project","roas"]
).properties(height=280)
st.altair_chart(roas_chart, use_container_width=True)

# 4) Revenue trend by month
st.subheader("Revenue Trend")
line_rev = alt.Chart(monthly).mark_line(point=True).encode(
    x=alt.X("month:T", title="Month"),
    y=alt.Y("revenue:Q", title="Revenue (€)"),
    color=alt.Color("project:N", legend=alt.Legend(title="Project")),
    tooltip=["month:T","project:N","revenue:Q"]
).properties(height=300)
st.altair_chart(line_rev, use_container_width=True)

# ===========================
# 🔥 NEW CHARTS YOU ASKED FOR
# ===========================

# A) Bounce Rate by Project (weighted bar)
st.subheader("Bounce Rate by Project (weighted)")
# session-weighted bounce rate per project = sum(sessions*bounce_rate)/sum(sessions)
proj_bounce = (fdf.groupby("project", as_index=False)
                 .agg(sessions=("sessions","sum"),
                      bounces=("bounces_est","sum")))
proj_bounce["bounce_rate_weighted"] = np.where(
    proj_bounce["sessions"]>0,
    proj_bounce["bounces"]/proj_bounce["sessions"],
    np.nan
)
bounce_chart = alt.Chart(proj_bounce).mark_bar().encode(
    x=alt.X("project:N", title="Project"),
    y=alt.Y("bounce_rate_weighted:Q", title="Bounce Rate", axis=alt.Axis(format='%')),
    tooltip=["project","bounce_rate_weighted"]
).properties(height=280)
st.altair_chart(bounce_chart, use_container_width=True)

# B) Session Count (pie chart) by Project
st.subheader("Sessions Share by Project (Pie)")
sessions_pie = proj_bounce[["project","sessions"]].copy()
pie_chart = alt.Chart(sessions_pie).mark_arc().encode(
    theta=alt.Theta(field="sessions", type="quantitative"),
    color=alt.Color(field="project", type="nominal", legend=alt.Legend(title="Project")),
    tooltip=["project","sessions"]
).properties(height=320)
st.altair_chart(pie_chart, use_container_width=True)

# C) Transactions vs Avg Session Time by Project (scatter)
st.subheader("Transactions vs Average Session Duration by Project")
# session-weighted average session duration (seconds) per project
def weighted_avg_duration(subdf: pd.DataFrame) -> float:
    # Use sessions as weights for averaging duration
    w = subdf["sessions"].values
    v = subdf["avg_session_duration_seconds"].values if "avg_session_duration_seconds" in subdf.columns else np.zeros_like(w)
    # guard against all-zero weights
    return float(np.average(v, weights=np.where(w>0, w, 1)))

proj_scatter = (fdf.groupby("project")
                  .apply(lambda g: pd.Series({
                      "transactions": g["transactions"].sum(),
                      "avg_session_seconds_weighted": weighted_avg_duration(g)
                  }))
                  .reset_index())

proj_scatter["avg_session_minutes"] = proj_scatter["avg_session_seconds_weighted"] / 60.0

scatter = alt.Chart(proj_scatter).mark_circle(size=140).encode(
    x=alt.X("avg_session_minutes:Q", title="Avg Session Duration (minutes)"),
    y=alt.Y("transactions:Q", title="Transactions"),
    color=alt.Color("project:N", legend=alt.Legend(title="Project")),
    tooltip=[
        alt.Tooltip("project:N", title="Project"),
        alt.Tooltip("avg_session_minutes:Q", title="Avg Session (min)", format=".2f"),
        alt.Tooltip("transactions:Q", title="Transactions", format=",.0f")
    ]
).properties(height=320)
st.altair_chart(scatter, use_container_width=True)

# Table preview
with st.expander("Preview filtered rows"):
    st.dataframe(fdf.sort_values("date").head(50))

