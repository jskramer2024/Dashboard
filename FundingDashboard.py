"""
Global Financial Markets Dashboard
IMF Monetary and Capital Markets Department
Funding & Financial Conditions Monitor

Navigation: Country Group → Country → Indicator Tab → Charts
Click on slope charts to update the cross-section curve below.

Run:  streamlit run FundingDashboard.py
Deps: pip install streamlit pandas plotly streamlit-plotly-events
Data: Run export_dashboard_data.m, then (optionally) Example_Factors.m in MATLAB first.
"""

from __future__ import annotations
import os, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

HAS_PLOTLY_EVENTS = False

# ─── Data directory ────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "dashboard_data"))

# ─── IMF Color palette ─────────────────────────────────────────────────────────
C = dict(
    navy   = "#003082",   # IMF primary blue
    blue   = "#009CDE",   # IMF accent blue
    red    = "#C8102E",   # IMF red
    green  = "#006B3F",   # IMF green
    gold   = "#FDB71A",   # IMF gold
    gray   = "#6B7280",
    lgray  = "#E5E7EB",
    border = "#D1D5DB",
    text   = "#111827",
    sub    = "#6B7280",
    orange = "#D97706",
    teal   = "#0F766E",
    purple = "#7C3AED",
)
CURRENCY_COLOR = dict(US=C["navy"], EA=C["green"], GB=C["red"], JP=C["gold"])
TENOR_COLORS   = [C["navy"], C["blue"], C["teal"], C["green"], C["gold"], C["red"]]
CHART_H = 500
XS_H = 500

# ─── Constants ─────────────────────────────────────────────────────────────────
G4 = ["US", "EA", "GB", "JP"]
GROUP_DISPLAY = {
    "G4":        "G4 Funding Markets",
    "G10":       "G10 Advanced",
    "EM Main":   "EM Main",
    "EM Other":  "EM Other",
    "Euro Area": "Euro Area",
    "AE Other":  "AE Other",
}
GROUP_COUNTRIES = {"G4": G4}
REPO_TENORS = ["1D", "1W", "1M", "3M", "6M", "1Y"]
REPO_TAU    = [1/252, 1/52, 1/12, 0.25, 0.5, 1.0]

# Key economic event markers
EVENTS = [
    ("2007-08-09", "BNP"),
    ("2008-09-15", "GFC"),
    ("2010-05-06", "EZ Crisis"),
    ("2012-07-26", "Draghi"),
    ("2020-03-23", "COVID"),
    ("2022-02-24", "Ukraine"),
    ("2022-03-16", "Fed+"),
    ("2023-03-10", "SVB"),
]

# Data availability notes per country/spread type
LOG_NOTES = {
    "EA": {
        "GC_OIS": "GC repo from Aug 2011 (ICAP PRSB).",
        "Xccy":   "Cross-currency basis from Aug 2023 (Haver).",
    },
    "GB": {
        "GC_OIS": "GC repo from Aug 2011; SC repo from Apr 2014.",
        "SC_GC":  "Special vs GC: DMO specials.",
    },
    "JP": {
        "GC_OIS": "JP GC repo only from Oct 2025.",
    },
    "US": {
        "GC_OIS": "GC 1D/1W from Feb 2004; 1M-1Y from Jun 2013.",
        "SC_GC":  "DTCC GCF UST vs GC overnight from Jan 2005.",
        "CP_OIS": "AA NonFin / A2P2 / AA ABCP vs 1M OIS.",
    },
}

# ─── Tenor helpers ─────────────────────────────────────────────────────────────
def parse_tenor_label(label: str) -> float:
    """Tenor string → decimal years. '1D'=1/252, '1W'=1/52, '3M'=0.25, '10Y'=10."""
    label = label.strip()
    try:
        if "Y" in label and "M" in label:
            y_part = int(label.split("Y")[0])
            m_part = int(label.split("Y")[1].replace("M", ""))
            return y_part + m_part / 12
        if label.endswith("D"): return int(label[:-1]) / 252
        if label.endswith("W"): return int(label[:-1]) / 52
        if label.endswith("M"): return int(label[:-1]) / 12
        if label.endswith("Y"): return float(label[:-1])
    except Exception:
        pass
    return float("nan")

# ─── Column name helpers — _1D / _ON fallback ──────────────────────────────────
def gc_ois_col(cty: str, tenor: str, df: pd.DataFrame) -> Optional[str]:
    """GC-OIS spread column; tries _1D then _ON for overnight."""
    c = f"{cty}_GC_OIS_{tenor}"
    if c in df.columns: return c
    if tenor == "1D" and f"{cty}_GC_OIS_ON" in df.columns: return f"{cty}_GC_OIS_ON"
    return None

def repo_gc_col(cty: str, tenor: str, df: pd.DataFrame) -> Optional[str]:
    """GC repo rate column; tries _1D then _ON for overnight."""
    c = f"{cty}_RepoGC_{tenor}"
    if c in df.columns: return c
    if tenor == "1D" and f"{cty}_RepoGC_ON" in df.columns: return f"{cty}_RepoGC_ON"
    return None

def ois_1d_col(cty: str, df: pd.DataFrame) -> Optional[str]:
    """OIS overnight column from rates.csv."""
    return f"{cty}_OIS1D" if f"{cty}_OIS1D" in df.columns else None

def cp_ois_cols(df: pd.DataFrame) -> dict[str, str]:
    """US CP-OIS columns; supports refreshed CP_OIS names and stale CPOIS names."""
    cols = {}
    for c in df.columns:
        if c.startswith("US_CP_OIS_"):
            cols[c] = c.replace("US_CP_OIS_", "")
        elif c.startswith("US_CPOIS_"):
            cols[c] = c.replace("US_CPOIS_", "")
    return cols

def special_repo_col(cty: str, df: pd.DataFrame) -> Optional[str]:
    if cty == "GB" and "GB_RepoSC_1D" in df.columns:
        return "GB_RepoSC_1D"
    if cty == "US" and "US_DTCC_UST_1D" in df.columns:
        return "US_DTCC_UST_1D"
    if cty == "US" and "US_DTCC_UST_ON" in df.columns:
        return "US_DTCC_UST_ON"
    return None

# ─── Chart style ───────────────────────────────────────────────────────────────
_EV_LINE = "rgba(110,110,110,0.28)"

AXIS = dict(
    showgrid=False,
    showline=True,  linecolor="black", linewidth=1,
    zeroline=False,
    ticks="outside", ticklen=5, tickwidth=1, tickcolor="black",
    tickfont=dict(size=13, color="black"),
    title_font=dict(size=11),
    title_standoff=8,
)
XAXIS = {**AXIS, "tickangle": -45}
YAXIS = {**AXIS, "tickangle": 0}
YAXIS_ZERO = {**YAXIS,
              "zeroline": True,
              "zerolinecolor": "rgba(100,100,100,0.30)",
              "zerolinewidth": 1}

def hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

def _layout(fig: go.Figure, height: int, extra_margin_r: int = 24):
    fig.update_layout(
        height=height,
        margin=dict(l=64, r=extra_margin_r, t=72, b=58),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.055,
                    xanchor="left", x=0, font=dict(size=9),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(15,15,15,0.90)",
                        bordercolor=C["border"],
                        font=dict(size=10, color="white")),
    )

def base_fig(height: int = CHART_H) -> go.Figure:
    """Single-axis figure with IMF chart style."""
    fig = go.Figure()
    _layout(fig, height)
    fig.update_xaxes(**XAXIS)
    fig.update_yaxes(**YAXIS)
    return fig

def dual_fig(height: int = CHART_H) -> go.Figure:
    """Dual-axis figure (secondary_y) with IMF chart style."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    _layout(fig, height, extra_margin_r=60)
    fig.update_xaxes(**XAXIS)
    fig.update_yaxes(**YAXIS, secondary_y=False)
    fig.update_yaxes(**YAXIS, secondary_y=True)
    return fig

def xs_fig(height: int = XS_H) -> go.Figure:
    """Cross-section (curve snapshot) figure."""
    return base_fig(height=height)

def add_events(fig: go.Figure, start_date=None):
    """Add vertical event markers to a time-series figure."""
    for date_str, evt in EVENTS:
        dt = pd.Timestamp(date_str)
        if start_date is not None and dt < pd.Timestamp(start_date):
            continue
        fig.add_shape(
            type="line",
            x0=dt, x1=dt, y0=0, y1=1,
            xref="x", yref="paper",
            line_width=1, line_dash="dot", line_color=_EV_LINE,
        )
        fig.add_annotation(
            x=dt, y=1, xref="x", yref="paper",
            text=evt, showarrow=False, textangle=-90,
            xanchor="left", yanchor="top", xshift=3, yshift=0,
            font=dict(size=8, color="rgba(100,100,100,0.60)"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        )

# ─── Trace helpers ─────────────────────────────────────────────────────────────
def line(fig, s: pd.Series, name: str, color: str,
         width: float = 2.0, dash=None, sy=None, showlegend=True):
    """Add a line trace. sy=True/False for secondary_y (dual_fig only)."""
    trace = go.Scatter(
        x=s.index, y=s.values, name=name, mode="lines",
        line=dict(color=color, width=width, dash=dash),
        showlegend=showlegend,
    )
    if sy is not None:
        fig.add_trace(trace, secondary_y=sy)
    else:
        fig.add_trace(trace)

def area(fig, s: pd.Series, name: str, color: str, width: float = 1.8):
    """Add a zero-filled area trace."""
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, name=name, mode="lines",
        fill="tozeroy", fillcolor=hex_to_rgba(color, 0.18),
        line=dict(color=color, width=width),
    ))
    fig.update_yaxes(**YAXIS_ZERO)

def hline_zero(fig):
    fig.add_hline(y=0, line_width=0.8,
                  line_color="rgba(100,100,100,0.30)", line_dash="solid")

def label(text: str):
    st.markdown(f'<p class="section-label">{text}</p>', unsafe_allow_html=True)

def note(text: str):
    st.markdown(f'<p class="chart-note">{text}</p>', unsafe_allow_html=True)

def chart(fig: go.Figure, key: str):
    fig.update_yaxes(title_text="")
    fig.update_layout(margin=dict(l=64, r=72, t=72, b=58))
    fig.update_xaxes(automargin=False)
    fig.update_yaxes(automargin=False, tickangle=0)
    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": False}, key=key,
    )

def clip(df: pd.DataFrame, s, e) -> pd.DataFrame:
    if df.empty: return df
    return df.loc[(df.index >= s) & (df.index <= e)]

def get_curve_at(df: pd.DataFrame, cols_parsed: list, ts: pd.Timestamp):
    """(x_years, y_vals) for a rate curve at timestamp ts using LOCF."""
    row = df.loc[:ts].tail(1)
    if row.empty: return [], []
    xvals, yvals = [], []
    for yrs, _, col in cols_parsed:
        if col in df.columns:
            v = float(row[col].iloc[-1])
            if not np.isnan(v):
                xvals.append(yrs)
                yvals.append(v)
    return xvals, yvals

def yoy(s: pd.Series, window: int = 252) -> pd.Series:
    """Year-on-year % change with rolling window fallback."""
    return s.pct_change(periods=window) * 100

def curve_points(df: pd.DataFrame, prefix: str, ts: pd.Timestamp, max_tau: float = None):
    """LOCF curve snapshot as sorted (tau, label, value) tuples."""
    if df.empty:
        return []
    row = df.loc[:ts].tail(1)
    if row.empty:
        return []
    pts = []
    for col in df.columns:
        if not col.startswith(prefix):
            continue
        lbl = col.replace(prefix, "")
        tau = parse_tenor_label(lbl)
        if np.isnan(tau) or (max_tau is not None and tau > max_tau):
            continue
        val = float(row[col].iloc[-1])
        if not np.isnan(val):
            pts.append((tau, lbl, val))
    return sorted(pts, key=lambda x: x[0])

def combined_curve_fig(height: int = XS_H) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    _layout(fig, height, extra_margin_r=72)
    fig.update_xaxes(**XAXIS)
    fig.update_yaxes(**YAXIS, secondary_y=False)
    fig.update_yaxes(**YAXIS, secondary_y=True)
    return fig

def mm_axis(fig: go.Figure):
    fig.update_xaxes(
        title_text="Maturity (years)", type="linear",
        tickvals=REPO_TAU, ticktext=REPO_TENORS,
        range=[0, 1.08], **XAXIS,
    )

def curve_axis(fig: go.Figure):
    fig.update_xaxes(
        title_text="Maturity (years)", type="linear",
        tickvals=[0.25, 1, 2, 5, 10, 20, 30],
        ticktext=["3M", "1Y", "2Y", "5Y", "10Y", "20Y", "30Y"],
        **XAXIS,
    )

def nearest_point(points: list, target: float):
    if not points:
        return None
    return min(points, key=lambda p: abs(p[0] - target))

# ─── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IMF | Global Financial Markets",
    page_icon="🏛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

  .imf-header {{
      border-left: 4px solid {C["navy"]};
      padding-left: 14px; margin-bottom: 4px;
  }}
  .imf-header h2 {{
      margin: 0 0 2px; font-weight: 700; letter-spacing: -0.02em;
      color: {C["navy"]};
  }}
  .imf-header p {{
      margin: 0; font-size: 0.82rem; opacity: 0.55;
  }}

  .metric-card {{
      background: var(--secondary-background-color);
      border: 1px solid rgba(128,128,128,0.18);
      border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
  }}
  .metric-label {{
      font-size: 0.67rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.08em; opacity: 0.50; margin-bottom: 3px;
      color: var(--text-color);
  }}
  .metric-value {{ font-size: 1.40rem; font-weight: 600; color: var(--text-color); }}
  .metric-pos   {{ color: {C["green"]}; font-size: 0.80rem; }}
  .metric-neg   {{ color: {C["red"]};   font-size: 0.80rem; }}

  .section-label {{
      font-size: 0.70rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.08em; opacity: 1; margin-bottom: 2px;
      color: {C["navy"]};
  }}
  .chart-note {{
      font-size: 0.70rem; opacity: 0.62; margin: 4px 0 10px 0;
      min-height: 1.15rem;
      font-style: italic; color: {C["gray"]};
  }}
  .data-flag {{
      font-size: 0.72rem; background: {hex_to_rgba(C["gold"], 0.22)};
      border: 1px solid {hex_to_rgba(C["gold"], 0.55)};
      border-radius: 4px; padding: 2px 8px; display: inline-block;
      color: var(--text-color);
  }}

  .stTabs [data-baseweb="tab-list"] {{ gap: 2px; }}
  .stTabs [data-baseweb="tab"] {{
      padding: 7px 16px; font-size: 0.83rem; font-weight: 500;
      border-radius: 6px 6px 0 0;
  }}
  div[data-testid="stRadio"] div[role="radiogroup"] {{
      flex-direction: row; gap: 6px; flex-wrap: wrap;
  }}
  hr {{ opacity: 0.20; margin: 10px 0; }}
</style>
""", unsafe_allow_html=True)

# ─── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    if not p.exists(): return pd.DataFrame()
    return pd.read_csv(p, parse_dates=["Date"], index_col="Date")

@st.cache_data(ttl=3600, show_spinner=False)
def load_meta() -> pd.DataFrame:
    p = DATA_DIR / "country_meta.csv"
    if not p.exists(): return pd.DataFrame()
    return pd.read_csv(p)

with st.spinner("Loading market data…"):
    meta     = load_meta()
    rates    = load("rates.csv")
    yields   = load("yields_key.csv")
    credit   = load("credit.csv")
    eqfx     = load("equity_fx.csv")
    ois_crv  = load("g4_ois_curves.csv")
    yld_crv  = load("g4_yield_curves.csv")
    repo_crv = load("g4_repo_curves.csv")
    spreads  = load("g4_spreads.csv")
    xccy     = load("g4_xccy.csv")
    vfci     = load("vfci_factors.csv")   # may be empty if not yet exported

if meta.empty:
    st.error("No data found in `dashboard_data/`. Run `export_dashboard_data.m` in MATLAB first.")
    st.stop()

HAS_VFCI = not vfci.empty

def metric_card(lbl: str, val: float, unit: str = "", delta: float = None):
    if np.isnan(val):
        return
    cls_d, sign, delta_html = "", "", ""
    if delta is not None and not np.isnan(delta):
        cls_d = "metric-pos" if delta >= 0 else "metric-neg"
        sign  = "+" if delta >= 0 else ""
        delta_html = f'<div style="margin-top:3px"><span class="{cls_d}">{sign}{delta:.1f}{unit} ytd</span></div>'
    col_v = C["red"] if unit == " bps" and val > 0 else (C["navy"] if unit == " bps" else "inherit")
    fmt_v = f"{val:+.1f}" if unit == " bps" else f"{val:.2f}"
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{lbl}</div>
      <div class="metric-value" style="color:{col_v}">{fmt_v}{unit}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)

def latest_value(df: pd.DataFrame, col: Optional[str]) -> float:
    if not col or df.empty or col not in df.columns:
        return np.nan
    s = df[col].dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan

def latest_series_value(s: Optional[pd.Series]) -> float:
    if s is None:
        return np.nan
    s = s.dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan

# ─── Header ────────────────────────────────────────────────────────────────────
top_l, top_r = st.columns([5, 1])
with top_l:
    last_date = spreads.index[-1].strftime("%d %b %Y") if not spreads.empty else (
        rates.index[-1].strftime("%d %b %Y") if not rates.empty else "—")
    st.markdown(f"""
    <div class="imf-header">
      <h2>Global Financial Markets Monitor</h2>
      <p>IMF Monetary and Capital Markets Department · Funding &amp; Financial Conditions</p>
    </div>""", unsafe_allow_html=True)
with top_r:
    st.markdown(
        f'<div style="text-align:right;opacity:0.45;font-size:0.74rem;padding-top:14px">'
        f'Data through<br><strong style="font-size:0.87rem;opacity:1">{last_date}</strong></div>',
        unsafe_allow_html=True)

st.divider()

# ─── Top controls ─────────────────────────────────────────────────────────────
ctl1, ctl2, ctl3 = st.columns([3, 3, 4])
with ctl1:
    label("Country Group")
    group_choice = st.radio(
        "grp", list(GROUP_DISPLAY.keys()),
        format_func=lambda k: GROUP_DISPLAY[k],
        horizontal=True, label_visibility="collapsed", key="group_radio",
        index=0,
    )
with ctl2:
    label("Sample Period")
    period = st.select_slider(
        "period",
        options=["1M", "3M", "6M", "1Y", "3Y", "5Y", "10Y", "15Y", "Max"],
        value="1Y",
        label_visibility="collapsed", key="period_slider",
    )
with ctl3:
    if group_choice == "G4":
        st.markdown(
            '<span class="data-flag">G4 view includes the extended funding-market detail.</span>',
            unsafe_allow_html=True,
        )
    if not HAS_VFCI:
        st.markdown(
            '<span class="data-flag">⚠ vfci_factors.csv not found — '
            'Financial Conditions tab uses proxy series. Run Example_Factors.m.</span>',
            unsafe_allow_html=True,
        )

today  = datetime.date.today()
end_dt = rates.index.max() if not rates.empty else pd.Timestamp(today)
_PERIOD_OFFSET = {
    "1M":  pd.DateOffset(months=1),
    "3M":  pd.DateOffset(months=3),
    "6M":  pd.DateOffset(months=6),
    "1Y":  pd.DateOffset(years=1),
    "3Y":  pd.DateOffset(years=3),
    "5Y":  pd.DateOffset(years=5),
    "10Y": pd.DateOffset(years=10),
    "15Y": pd.DateOffset(years=15),
    "Max": pd.DateOffset(years=40),
}
start = end_dt - _PERIOD_OFFSET[period]
end   = end_dt

# ─── Countries for selected group ─────────────────────────────────────────────
group_ctys = GROUP_COUNTRIES.get(
    group_choice,
    meta[meta["Group"] == group_choice]["Code"].tolist() if not meta.empty else G4,
)

if not group_ctys:
    st.info("No countries in this group.")
    st.stop()

# ─── Country tabs ─────────────────────────────────────────────────────────────
cty_labels = []
for c in group_ctys:
    row = meta[meta["Code"] == c]
    cty_labels.append(
        f"{c}  {row['Currency'].iloc[0].replace(' Curncy','')}" if not row.empty else c
    )

country_tabs = st.tabs(cty_labels)

for tab_i, (cty_tab, cty) in enumerate(zip(country_tabs, group_ctys)):
    with cty_tab:
        is_g4    = cty in G4
        has_pr   = f"{cty}_PR"    in rates.columns
        has_ois  = ois_1d_col(cty, rates) is not None
        has_yld  = f"{cty}_Yld2Y" in yields.columns
        has_ig   = f"{cty}_IG"    in credit.columns
        has_hy   = f"{cty}_HY"    in credit.columns
        has_sov  = f"{cty}_EMSov" in credit.columns
        has_eq   = f"{cty}_EQ"    in eqfx.columns
        show_funding_detail = group_choice == "G4"
        has_repo = show_funding_detail and is_g4 and not repo_crv.empty
        has_xccy = show_funding_detail and is_g4 and cty != "US" and not xccy.empty

        # ── Build indicator tab list ──────────────────────────────────────────
        ind_names = ["📊 Financial Conditions"]   # always shown
        if has_pr or has_ois:
            ind_names.append("📈 Rates & OIS")
        if has_repo:
            ind_names.append("💧 Funding & Repo")
        if has_xccy:
            ind_names.append("🌐 Cross-Currency")

        ind_tabs = st.tabs(ind_names)
        tm = {n: t for n, t in zip(ind_names, ind_tabs)}

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — Financial Conditions (5 dual-axis charts)
        # ══════════════════════════════════════════════════════════════════════
        with tm["📊 Financial Conditions"]:
            rc = clip(rates,   start, end)
            sc = clip(spreads, start, end) if is_g4 else pd.DataFrame()
            cc = clip(credit,  start, end)
            ec = clip(eqfx,    start, end)
            yc = clip(yields,  start, end)
            vc = clip(vfci,    start, end) if HAS_VFCI else pd.DataFrame()

            COLOR_L = CURRENCY_COLOR.get(cty, C["navy"])
            COLOR_R = C["blue"]

            # Helper: pull a VFCI column, return series or None
            def vf(col_name: str) -> Optional[pd.Series]:
                names = [col_name]
                if col_name == "SWAP_Sprd":
                    names.append("SWAP")
                for name in names:
                    full = f"{cty}_{name}"
                    if not vc.empty and full in vc.columns:
                        s = vc[full].dropna()
                        return s if not s.empty else None
                return None

            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                metric_card("LIQ", latest_series_value(vf("LIQ")), unit=" bps")
            with fc2:
                metric_card("TERM", latest_series_value(vf("TERM")), unit=" bps")
            with fc3:
                metric_card("SWAP_Sprd", latest_series_value(vf("SWAP_Sprd")), unit=" bps")
            with fc4:
                metric_card("FX Y/Y", latest_series_value(vf("FX_Ret")), unit="%")

            # ── Chart 1: LIQ (LHS) + Policy Rate (RHS) ──────────────────────
            row1_l, row1_r = st.columns(2)
            with row1_l:
                label("Liquidity Premium & Policy Rate")
                fig1 = dual_fig()
                liq = vf("LIQ")
                pr_s = rc[f"{cty}_PR"].dropna() if has_pr else None
                if liq is not None:
                    line(fig1, liq, "Liquidity (LHS)", COLOR_L, width=2.0, sy=False)
                    y1_title = "LIQ factor (bps)"
                elif is_g4 and f"{cty}_OIS_PR" in sc.columns:
                    line(fig1, sc[f"{cty}_OIS_PR"].dropna(),
                         "OIS–PR proxy (LHS)", COLOR_L, width=2.0, sy=False)
                    y1_title = "OIS–Policy Rate (bps)"
                else:
                    y1_title = "—"
                if pr_s is not None:
                    line(fig1, pr_s, "Policy Rate (RHS)", C["red"], width=1.5, dash="dash", sy=True)
                add_events(fig1, start)
                fig1.update_yaxes(title_text=y1_title, secondary_y=False, **YAXIS_ZERO)
                fig1.update_yaxes(title_text="Policy Rate (%)",
                                  secondary_y=True, **YAXIS)
                chart(fig1, f"fc_liq_{cty}")
                note("LIQ is the short-end liquidity premium: it is defined as 3M market/OIS rate relative to the overnight or policy-rate anchor, with the latter shown on the right axis.")
                if liq is None:
                    note("Fallback: OIS–PR spread used (vfci_factors.csv not loaded)")

            # ── Chart 2: TERM spread (LHS) + Swap Spread (RHS) ───────────────
            with row1_r:
                label("Term Factor & Swap Spread")
                fig2 = dual_fig()
                term  = vf("TERM")
                swap_ = vf("SWAP_Sprd")
                if term is not None:
                    line(fig2, term, "TERM: 10Y-3M target (LHS)", COLOR_L, width=2.0, sy=False)
                    t2_lhs = "TERM factor (bps)"
                elif has_yld and f"{cty}_Yld10Y" in yc.columns and f"{cty}_Yld2Y" in yc.columns:
                    ts_fb = (yc[f"{cty}_Yld10Y"] - yc[f"{cty}_Yld2Y"]) * 100
                    line(fig2, ts_fb.dropna(), "10Y–2Y (LHS)", COLOR_L, width=2.0, sy=False)
                    t2_lhs = "10Y–2Y yield spread (bps)"
                else:
                    t2_lhs = "—"
                if swap_ is not None:
                    line(fig2, swap_, "Swap Spread (RHS)", C["red"], width=1.5, dash="dash", sy=True)
                    t2_rhs = "SWAP_Sprd (bps)"
                elif is_g4 and f"{cty}_OIS_Slope" in sc.columns:
                    line(fig2, sc[f"{cty}_OIS_Slope"].dropna(),
                         "OIS slope 3M–1D (RHS)", C["red"], width=1.5, dash="dash", sy=True)
                    t2_rhs = "OIS 3M–1D slope (bps)"
                else:
                    t2_rhs = "—"
                add_events(fig2, start)
                hline_zero(fig2)
                fig2.update_yaxes(title_text=t2_lhs, secondary_y=False, **YAXIS_ZERO)
                fig2.update_yaxes(title_text=t2_rhs, secondary_y=True, **YAXIS_ZERO)
                chart(fig2, f"fc_term_{cty}")
                note("TERM is the rate differential between 10y and 3M tenors, based on swap rates with yield fallback/basis adjustment. SWAP_Sprd is the 10y swap rate less the same-maturity government yield.")
                if cty == "US" and term is not None and swap_ is not None:
                    term_latest = term.dropna().iloc[-1] if not term.dropna().empty else np.nan
                    swap_latest = swap_.dropna().iloc[-1] if not swap_.dropna().empty else np.nan
                    note(f"US factor columns: US_TERM = {term_latest:.4f}; US_SWAP_Sprd = {swap_latest:.4f}.")

            row2_l, row2_r = st.columns(2)

            # ── Chart 3: IG/HY spreads (LHS) + VOL (RHS) ─────────────────────
            with row2_l:
                label("Credit Spreads & Equity Volatility")
                fig3 = dual_fig()
                ig_f   = vf("IG_Corp")
                hy_f   = vf("HY_Corp")
                vol_f  = vf("VOL")
                ig_raw = cc[f"{cty}_IG"].dropna()  if has_ig  else None
                hy_raw = cc[f"{cty}_HY"].dropna()  if has_hy  else None
                vol_raw = ec[f"{cty}_VOL"].dropna() if has_eq  else None
                if ig_f is not None:
                    line(fig3, ig_f, "IG Corp (LHS)", COLOR_L, width=2.0, sy=False)
                elif ig_raw is not None and not ig_raw.empty:
                    line(fig3, ig_raw, "IG OAS (LHS)", COLOR_L, width=2.0, sy=False)
                if hy_f is not None:
                    line(fig3, hy_f, "HY Corp (LHS)", C["orange"], width=1.5, sy=False)
                elif hy_raw is not None and not hy_raw.empty:
                    line(fig3, hy_raw, "HY OAS (LHS)", C["orange"], width=1.5, sy=False)
                if vol_f is not None:
                    line(fig3, vol_f, "Volatility (RHS)", C["purple"], width=1.5, dash="dash", sy=True)
                elif vol_raw is not None and not vol_raw.empty:
                    line(fig3, vol_raw, "Implied vol (RHS)", C["purple"], width=1.5, dash="dash", sy=True)
                add_events(fig3, start)
                fig3.update_yaxes(title_text="Spread (bps)",
                                  secondary_y=False, **YAXIS)
                fig3.update_yaxes(title_text="Volatility (%)", secondary_y=True, **YAXIS)
                chart(fig3, f"fc_credit_vol_{cty}")
                note("Credit spreads show corporate risk premia on the left axis; implied equity volatility is shown on the right axis.")

            # ── Chart 4: EQ Y/Y % (LHS) + FX Y/Y % (RHS) ────────────────────
            with row2_r:
                label("Equity & FX Returns (year-on-year)")
                fig4 = dual_fig()
                eq_f  = vf("EQ_Ret")
                fx_f  = vf("FX_Ret")
                eq_raw = eqfx[f"{cty}_EQ"].dropna() if has_eq else None
                fx_raw = eqfx[f"{cty}_FX"].dropna() if f"{cty}_FX" in eqfx.columns else None
                if eq_f is not None:
                    line(fig4, eq_f, "EQ Y/Y % (LHS)", COLOR_L, width=2.0, sy=False)
                elif eq_raw is not None and len(eq_raw) > 252:
                    line(fig4, clip(yoy(eq_raw), start, end).dropna(), "EQ Y/Y % (LHS)", COLOR_L, width=2.0, sy=False)
                if fx_f is not None:
                    line(fig4, fx_f, "FX Y/Y % (RHS)", C["red"], width=1.5, dash="dash", sy=True)
                elif fx_raw is not None and len(fx_raw) > 252:
                    line(fig4, clip(yoy(fx_raw), start, end).dropna(), "FX Y/Y % (RHS)", C["red"], width=1.5, dash="dash", sy=True)
                add_events(fig4, start)
                hline_zero(fig4)
                fig4.update_yaxes(title_text="EQ return (%)",
                                  secondary_y=False, **YAXIS_ZERO)
                fig4.update_yaxes(title_text="FX return (%)",
                                  secondary_y=True,  **YAXIS_ZERO)
                chart(fig4, f"fc_eq_fx_{cty}")
                note("FX return is the one-year arithmetic return of the trade-weighted FX basket with positive returns showing domestic currency appreciation.")


            # ── Chart 5: Local sov spread (LHS) + HC sov spread (RHS) ────────
            show_sov_chart = group_choice in ["EM Main", "EM Other"]
            if show_sov_chart:
                label("EM Sovereign Spreads — Local Currency & Hard Currency")
            hc_f   = vf("HC_Sov")
            sov_disp = (vc["EA_SOV_Disp"].dropna()
                        if (HAS_VFCI and "EA_SOV_Disp" in vc.columns) else None)
            local_f = vf("SWAP_Sprd")    # local = swap spread as proxy for local sov basis
            sov_raw = cc[f"{cty}_EMSov"].dropna() if has_sov else None
            ig_r2   = cc[f"{cty}_IG"].dropna()    if has_ig  else None

            fig5 = dual_fig(420)
            # LHS: local currency sovereign risk
            if cty == "EA" and sov_disp is not None:
                line(fig5, sov_disp, "EA Sov Dispersion (LHS)", COLOR_L, width=2.0, sy=False)
                lhs5 = "EA sov dispersion (bps)"
            elif sov_raw is not None and not sov_raw.empty:
                line(fig5, sov_raw, "EM Sov spread (LHS)", COLOR_L, width=2.0, sy=False)
                lhs5 = "EM sov spread (bps)"
            else:
                lhs5 = "—"
            # RHS: hard currency sovereign risk
            if hc_f is not None:
                line(fig5, hc_f, "HC Sov spread (RHS)", C["red"], width=1.5, dash="dash", sy=True)
                rhs5 = "HC sov spread (bps)"
            elif sov_raw is not None and not sov_raw.empty:
                line(fig5, sov_raw, "EM Sov (EMBIG) (RHS)", C["red"], width=1.5, dash="dash", sy=True)
                rhs5 = "EMBIG stripped spread (bps)"
            else:
                rhs5 = "—"
            add_events(fig5, start)
            fig5.update_yaxes(title_text=lhs5,
                              secondary_y=False, **YAXIS)
            fig5.update_yaxes(title_text=rhs5,
                              secondary_y=True,  **YAXIS)
            if show_sov_chart:
                chart(fig5, f"fc_sov_{cty}")

            note("Sources: Bloomberg L.P.; ICE BofA; JPMorgan; Haver Analytics; IMF staff calculations.")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — Rates & OIS
        # ══════════════════════════════════════════════════════════════════════
        if "📈 Rates & OIS" in tm:
            with tm["📈 Rates & OIS"]:
                r_rates   = clip(rates,   start, end)
                r_spreads = clip(spreads, start, end) if is_g4 else pd.DataFrame()
                color     = CURRENCY_COLOR.get(cty, C["navy"])
                pr_col    = f"{cty}_PR"
                ois_col   = ois_1d_col(cty, rates)

                # Metric row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    if has_pr:
                        metric_card("Policy Rate", float(rates[pr_col].dropna().iloc[-1]), unit="%")
                with m2:
                    if ois_col:
                        metric_card("OIS Overnight", float(rates[ois_col].dropna().iloc[-1]), unit="%")
                with m3:
                    if is_g4 and f"{cty}_OIS_PR" in spreads.columns:
                        metric_card("OIS–Policy Spread",
                                    float(spreads[f"{cty}_OIS_PR"].dropna().iloc[-1]), unit=" bps")
                with m4:
                    col_gcon = gc_ois_col(cty, "1D", spreads) if is_g4 else None
                    if col_gcon:
                        metric_card("GC–OIS (1D)", float(spreads[col_gcon].dropna().iloc[-1]), unit=" bps")

                col_l, col_r = st.columns(2)

                # Level chart
                with col_l:
                    label("Policy Rate · OIS Overnight · GC Repo 1D · SC Repo 1D  (percent)")
                    fig = base_fig(height=CHART_H)
                    if has_pr:
                        line(fig, r_rates[pr_col].dropna(), "Policy Rate", color, width=2.0)
                    if ois_col:
                        line(fig, r_rates[ois_col].dropna(), "OIS overnight", C["blue"], dash="dash")
                    col_gc = repo_gc_col(cty, "1D", repo_crv) if is_g4 else None
                    if col_gc:
                        r_repo = clip(repo_crv, start, end)
                        line(fig, r_repo[col_gc].dropna(), "GC repo 1D",
                             color, dash="dot", width=1.2)
                    col_sc_repo = special_repo_col(cty, repo_crv) if is_g4 else None
                    if col_sc_repo:
                        r_repo = clip(repo_crv, start, end)
                        line(fig, r_repo[col_sc_repo].dropna(), "SC repo 1D",
                             C["purple"], dash="dashdot", width=1.2)
                    add_events(fig, start)
                    chart(fig, f"rates_level_{cty}")

                # OIS–PR spread
                with col_r:
                    label("OIS and GC Repo Slopes  (3M minus 1D, basis points)")
                    fig2 = base_fig(height=CHART_H)
                    if is_g4 and f"{cty}_OIS_Slope" in r_spreads.columns:
                        line(fig2, r_spreads[f"{cty}_OIS_Slope"].dropna(),
                             "OIS 3M-1D", C["blue"], width=2.0)
                    col_gc1 = repo_gc_col(cty, "1D", repo_crv) if is_g4 else None
                    col_gc3 = repo_gc_col(cty, "3M", repo_crv) if is_g4 else None
                    if col_gc1 and col_gc3:
                        r_repo = clip(repo_crv, start, end)
                        gc_slope = (r_repo[col_gc3] - r_repo[col_gc1]) * 100
                        line(fig2, gc_slope.dropna(), "GC repo 3M-1D", C["green"], width=2.0)
                    if is_g4 and f"{cty}_RepoSlope" in r_spreads.columns:
                        area(fig2, r_spreads[f"{cty}_RepoSlope"].dropna(),
                             "GC-OIS slope spread", C["red"], width=1.5)
                    add_events(fig2, start)
                    chart(fig2, f"rates_slopes_{cty}")

                if is_g4:
                    st.divider()
                    rates_period_idx = clip(spreads, start, end).index if not spreads.empty else pd.DatetimeIndex([])
                    all_idx = rates_period_idx if len(rates_period_idx) else (spreads.index if not spreads.empty else repo_crv.index)
                    min_r_date = all_idx[0].date()
                    max_r_date = all_idx[-1].date()
                    default_r_xs = st.session_state.get(f"rates_xs_{cty}", max_r_date)
                    default_r_xs = min(max(default_r_xs, min_r_date), max_r_date)
                    label("OIS/GC Curve Date")
                    rates_xs = st.slider(
                        "rates_xs_slider",
                        min_value=min_r_date, max_value=max_r_date,
                        value=default_r_xs, format="YYYY-MM-DD",
                        label_visibility="collapsed", key=f"rates_xs_slider_{cty}",
                    )
                    note(f"Selected date: {rates_xs:%d %b %Y}")
                    st.session_state[f"rates_xs_{cty}"] = rates_xs
                    rates_ts = pd.Timestamp(rates_xs)
                    rates_lbl = rates_ts.strftime("%d %b %Y")
                    label(f"OIS and GC Repo Curves as at {rates_lbl}")
                    fig_og = combined_curve_fig()
                    ois_pts = curve_points(ois_crv, f"{cty}_OIS_", rates_ts, max_tau=1.01)
                    repo_row = repo_crv.loc[:rates_ts].tail(1) if not repo_crv.empty else pd.DataFrame()
                    sp_row = spreads.loc[:rates_ts].tail(1) if not spreads.empty else pd.DataFrame()
                    if ois_pts:
                        fig_og.add_trace(go.Scatter(
                            x=[p[0] for p in ois_pts], y=[p[2] for p in ois_pts],
                            name="OIS/Swap", mode="lines+markers",
                            line=dict(color=C["navy"], width=2.2), marker=dict(size=6),
                        ), secondary_y=False)
                    rx, ry, bx, by = [], [], [], []
                    for tn, tau in zip(REPO_TENORS, REPO_TAU):
                        col_rg = repo_gc_col(cty, tn, repo_crv)
                        col_sp = gc_ois_col(cty, tn, spreads)
                        if col_rg and not repo_row.empty:
                            v = float(repo_row[col_rg].iloc[-1])
                            if not np.isnan(v):
                                rx.append(tau); ry.append(v)
                        if col_sp and not sp_row.empty:
                            v = float(sp_row[col_sp].iloc[-1])
                            if not np.isnan(v):
                                bx.append(tau); by.append(v)
                    if rx:
                        fig_og.add_trace(go.Scatter(
                            x=rx, y=ry, name="GC Repo", mode="lines+markers",
                            line=dict(color=C["green"], width=2.2, dash="dash"), marker=dict(size=6),
                        ), secondary_y=False)
                    if bx:
                        fig_og.add_trace(go.Bar(
                            x=bx, y=by, name="GC Repo - OIS",
                            width=[0.035] * len(bx),
                            marker_color=hex_to_rgba(C["red"], 0.45), opacity=0.80,
                        ), secondary_y=True)
                    hline_zero(fig_og)
                    mm_axis(fig_og)
                    fig_og.update_yaxes(title_text="Rate (%)", secondary_y=False)
                    fig_og.update_yaxes(title_text="GC - OIS (bps)",
                                        secondary_y=True, **YAXIS_ZERO)
                    chart(fig_og, f"rates_ois_gc_curve_{cty}")

                # Slope row + cross-section (G4 only)
                if False and is_g4:
                    st.divider()
                    label("Term slopes  (3M–1D differential, basis points)")
                    col_s1, col_s2 = st.columns(2)

                    with col_s1:
                        label("OIS slope (3M minus overnight)")
                        fig_s1 = base_fig(height=XS_H)
                        sc_ois = f"{cty}_OIS_Slope"
                        if sc_ois in r_spreads.columns:
                            area(fig_s1, r_spreads[sc_ois].dropna(), "OIS slope 3M/1D", color)
                        add_events(fig_s1, start)
                        if HAS_PLOTLY_EVENTS:
                            clicked_ois = plotly_events(fig_s1, click_event=True,
                                                        key=f"click_ois_{cty}")
                        else:
                            chart(fig_s1, f"rates_legacy_ois_slope_{cty}")
                            clicked_ois = []

                    with col_s2:
                        label("Repo spread slope: (GC–OIS 3M) − (GC–OIS 1D)")
                        fig_s2 = base_fig(height=XS_H)
                        sc_rp = f"{cty}_RepoSlope"
                        if sc_rp in r_spreads.columns:
                            area(fig_s2, r_spreads[sc_rp].dropna(), "Repo spread slope", C["red"])
                        add_events(fig_s2, start)
                        if HAS_PLOTLY_EVENTS:
                            clicked_repo = plotly_events(fig_s2, click_event=True,
                                                         key=f"click_repo_{cty}")
                        else:
                            chart(fig_s2, f"rates_legacy_repo_slope_{cty}")
                            clicked_repo = []

                    # Resolve clicked date
                    clicked_date = None
                    if HAS_PLOTLY_EVENTS:
                        for src in [clicked_ois, clicked_repo]:
                            if src:
                                try: clicked_date = pd.Timestamp(src[0]["x"])
                                except Exception: pass
                                break
                        if clicked_date is not None:
                            st.session_state[f"xs_{cty}"] = clicked_date.date()

                    # Cross-section date selector
                    st.divider()
                    col_cs_ctrl, _ = st.columns([2, 3])
                    with col_cs_ctrl:
                        label("Cross-section date")
                        all_idx = spreads.index if not spreads.empty else rates.index
                        default_xs = st.session_state.get(
                            f"xs_{cty}",
                            all_idx[-1].date() if len(all_idx) else end.date(),
                        )
                        xs_date = st.date_input(
                            "xs", value=default_xs,
                            min_value=all_idx[0].date(), max_value=all_idx[-1].date(),
                            label_visibility="collapsed", key=f"xs_{cty}",
                        )
                    if clicked_date:
                        st.caption("↑ Click any point above to jump to that date.")

                    xs_ts = pd.Timestamp(xs_date)
                    xs_lbl = xs_ts.strftime("%d %b %Y")
                    label(f"Rate curves at {xs_lbl}")
                    col_oc, col_sc_sp = st.columns(2)

                    # OIS curve cross-section
                    ois_cols_parsed = sorted(
                        [(parse_tenor_label(c.replace(f"{cty}_OIS_", "")),
                          c.replace(f"{cty}_OIS_", ""), c)
                         for c in ois_crv.columns if c.startswith(f"{cty}_OIS_")
                         and not np.isnan(parse_tenor_label(c.replace(f"{cty}_OIS_", "")))]
                    )
                    with col_oc:
                        fig_oc = xs_fig()
                        xv, yv = get_curve_at(ois_crv, ois_cols_parsed, xs_ts)
                        if xv:
                            fig_oc.add_trace(go.Scatter(
                                x=xv, y=yv, name="OIS curve",
                                mode="lines+markers",
                                line=dict(color=C["navy"], width=2.0),
                                marker=dict(size=5, color=C["navy"]),
                            ))
                        fig_oc.update_xaxes(
                            title_text="Maturity (years)", type="log",
                            tickvals=[1/252, 1/52, 1/12, 0.25, 0.5, 1, 2, 5, 10, 30],
                            ticktext=["1D","1W","1M","3M","6M","1Y","2Y","5Y","10Y","30Y"],
                            **XAXIS,
                        )
                        fig_oc.update_yaxes(title_text="Rate (%)")
                        fig_oc.update_layout(showlegend=False,
                            title=dict(text="OIS / Swap Rate Curve",
                                       font=dict(size=11, color=C["sub"])))
                        chart(fig_oc, f"rates_legacy_ois_curve_{cty}")

                    # GC-OIS spread curve cross-section (capped at 1Y)
                    with col_sc_sp:
                        fig_gc = xs_fig()
                        sp_x, sp_y = [], []
                        for tn, yr in zip(REPO_TENORS, REPO_TAU):
                            col_s = gc_ois_col(cty, tn, spreads)
                            if col_s:
                                row_s = spreads.loc[:xs_ts].tail(1)
                                if not row_s.empty:
                                    v = float(row_s[col_s].iloc[-1])
                                    if not np.isnan(v):
                                        sp_x.append(yr); sp_y.append(v)
                        if sp_x:
                            fig_gc.add_trace(go.Scatter(
                                x=sp_x, y=sp_y, name="GC–OIS",
                                mode="lines+markers",
                                line=dict(color=C["red"], width=2.0),
                                marker=dict(size=5, color=C["red"]),
                                fill="tozeroy",
                                fillcolor=hex_to_rgba(C["red"], 0.12),
                            ))
                        hline_zero(fig_gc)
                        fig_gc.update_xaxes(
                            title_text="Maturity (years)", type="log",
                            tickvals=REPO_TAU, ticktext=REPO_TENORS,
                            range=[np.log10(1/252 * 0.8), np.log10(1.25)],
                            **XAXIS,
                        )
                        fig_gc.update_yaxes(title_text="Spread (bps)")
                        fig_gc.update_layout(showlegend=False,
                            title=dict(text="GC Repo – OIS Spread Curve",
                                       font=dict(size=11, color=C["sub"])))
                        chart(fig_gc, f"rates_legacy_gc_curve_{cty}")

                note("Sources: Bloomberg L.P.; Haver Analytics. "
                     "OIS = overnight indexed swap (Swap curve, 1D column). "
                     "GC = general collateral repo. Spreads in basis points (bps).")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3 — Funding & Repo
        # ══════════════════════════════════════════════════════════════════════
        if "💧 Funding & Repo" in tm:
            with tm["💧 Funding & Repo"]:
                r_repo    = clip(repo_crv, start, end)
                r_spreads = clip(spreads,  start, end)
                color     = CURRENCY_COLOR.get(cty, C["navy"])
                gc_note   = LOG_NOTES.get(cty, {}).get("GC_OIS", "")

                f1, f2, f3, f4 = st.columns(4)
                with f1:
                    cp_cols_now = cp_ois_cols(r_spreads) if cty == "US" else {}
                    first_cp = next(iter(cp_cols_now.keys()), None)
                    metric_card("CP-OIS 1M", latest_value(r_spreads, first_cp), unit=" bps")
                with f2:
                    metric_card("GC Repo 1D", latest_value(r_repo, repo_gc_col(cty, "1D", r_repo)), unit="%")
                with f3:
                    metric_card("GC-OIS 1D", latest_value(r_spreads, gc_ois_col(cty, "1D", r_spreads)), unit=" bps")
                with f4:
                    metric_card("SC-GC 1D", latest_value(r_spreads, f"{cty}_SC_GC"), unit=" bps")

                if cty == "US":
                    cp_cols = cp_ois_cols(r_spreads)
                    if cp_cols:
                        label("Commercial Paper - OIS 1M Spreads  (basis points)")
                        note(LOG_NOTES.get("US", {}).get("CP_OIS", ""))
                        fig_cp = base_fig(height=CHART_H)
                        cp_colors = [C["navy"], C["red"], C["orange"]]
                        for k, (col_cp, lbl_cp) in enumerate(cp_cols.items()):
                            line(fig_cp, r_spreads[col_cp].dropna(),
                                 lbl_cp, cp_colors[k % 3], width=1.8)
                        hline_zero(fig_cp)
                        add_events(fig_cp, start)
                        chart(fig_cp, f"funding_cp_top_{cty}")
                        st.divider()

                # GC repo level
                label("GC Repo Rates by Tenor  (percent)")
                fig_rl = base_fig(height=CHART_H)
                for k, tn in enumerate(REPO_TENORS):
                    col_rg = repo_gc_col(cty, tn, r_repo)
                    if col_rg:
                        line(fig_rl, r_repo[col_rg].dropna(), tn,
                             TENOR_COLORS[k], width=1.8)
                add_events(fig_rl, start)
                chart(fig_rl, f"funding_gc_levels_{cty}")
                if gc_note: note(gc_note)

                # SC–GC spread
                sc_col = f"{cty}_SC_GC"
                if sc_col in r_spreads.columns:
                    st.divider()
                    sc_lbl = {"GB": "DMO specials vs GC",
                              "US": "DTCC UST vs GC overnight"}.get(cty, "SC–GC")
                    label(f"Special Collateral – GC Overnight  (bps) — {sc_lbl}")
                    fig_sc = base_fig(height=CHART_H)
                    area(fig_sc, r_spreads[sc_col].dropna(), "SC–GC", C["purple"])
                    add_events(fig_sc, start)
                    chart(fig_sc, f"funding_sc_gc_{cty}")
                    sc_note = LOG_NOTES.get(cty, {}).get("SC_GC", "")
                    if sc_note: note(sc_note)

                # CP–OIS (US only)
                if False and cty == "US":
                    cp_cols = cp_ois_cols(r_spreads)
                    if cp_cols:
                        st.divider()
                        label("Commercial Paper – OIS 1M Spreads  (basis points)")
                        note(LOG_NOTES.get("US", {}).get("CP_OIS", ""))
                        fig_cp = base_fig(height=CHART_H)
                        cp_colors = [C["navy"], C["red"], C["orange"]]
                        for k, (col_cp, lbl_cp) in enumerate(cp_cols.items()):
                            line(fig_cp, r_spreads[col_cp].dropna(),
                                 lbl_cp, cp_colors[k % 3], width=1.8)
                        hline_zero(fig_cp)
                        add_events(fig_cp, start)
                        chart(fig_cp, f"funding_cp_legacy_{cty}")


                note("Sources: Bloomberg L.P.; Haver Analytics; OFR; DTCC. "
                     "GC = general collateral. SC = special collateral. "
                     "Spreads = repo minus matched-maturity OIS. "
                     + gc_note)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4 — Yield Curve
        # ══════════════════════════════════════════════════════════════════════
        if "📉 Yield Curve" in tm:
            with tm["📉 Yield Curve"]:
                r_ylds = clip(yields,  start, end)
                r_ycrv = clip(yld_crv, start, end)

                label("Key Tenor Yields  (percent)")
                col_ya, col_yb = st.columns(2)
                with col_ya:
                    fig_yt = base_fig(height=CHART_H)
                    for k, tn in enumerate(["2Y","5Y","10Y"]):
                        col_y = f"{cty}_Yld{tn}"
                        if col_y in r_ylds.columns:
                            line(fig_yt, r_ylds[col_y].dropna(), tn,
                                 TENOR_COLORS[k*2], width=2.0)
                    add_events(fig_yt, start)
                    chart(fig_yt, f"yield_key_{cty}")

                with col_yb:
                    label("Term Spread 10Y – 2Y  (basis points)")
                    fig_ts = base_fig(height=CHART_H)
                    c10 = f"{cty}_Yld10Y"; c2 = f"{cty}_Yld2Y"
                    if c10 in r_ylds.columns and c2 in r_ylds.columns:
                        ts = (r_ylds[c10] - r_ylds[c2]) * 100
                        area(fig_ts, ts.dropna(), "10Y–2Y", CURRENCY_COLOR.get(cty, C["teal"]))
                    add_events(fig_ts, start)
                    chart(fig_ts, f"yield_term_{cty}")

                # Full yield curve cross-section
                ycrv_parsed = sorted(
                    [(parse_tenor_label(c.replace(f"{cty}_Yld_", "")),
                      c.replace(f"{cty}_Yld_", ""), c)
                     for c in yld_crv.columns if c.startswith(f"{cty}_Yld_")
                     and not np.isnan(parse_tenor_label(c.replace(f"{cty}_Yld_", "")))]
                )
                if ycrv_parsed:
                    st.divider()
                    label("Yield Curve Cross-Section")
                    col_ys, _ = st.columns([2, 3])
                    with col_ys:
                        xs3 = st.date_input(
                            "yxs", value=yld_crv.index[-1].date(),
                            min_value=yld_crv.index[0].date(),
                            max_value=yld_crv.index[-1].date(),
                            label_visibility="collapsed", key=f"yxs_{cty}",
                        )
                    fig_yc = xs_fig()
                    xs3_ts = pd.Timestamp(xs3)
                    xv, yv = get_curve_at(yld_crv, ycrv_parsed, xs3_ts)
                    if xv:
                        fig_yc.add_trace(go.Scatter(
                            x=xv, y=yv, mode="lines+markers",
                            line=dict(color=C["navy"], width=2.0),
                            marker=dict(size=5),
                            name=xs3_ts.strftime("%d %b %Y"),
                        ))
                    fig_yc.update_xaxes(title_text="Maturity (years)", **XAXIS)
                    fig_yc.update_yaxes(title_text="Yield (%)")
                    fig_yc.update_layout(showlegend=False)
                    chart(fig_yc, f"yield_curve_{cty}")

                note("Sources: Bloomberg L.P.; LSEG. Government benchmark yields.")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 5 — Credit
        # ══════════════════════════════════════════════════════════════════════
        if "💳 Credit" in tm:
            with tm["💳 Credit"]:
                r_cred = clip(credit, start, end)

                label("Credit Spreads  (basis points, option-adjusted)")
                fig_cr = base_fig(height=CHART_H)
                if f"{cty}_IG"    in r_cred.columns:
                    line(fig_cr, r_cred[f"{cty}_IG"].dropna(),    "IG OAS",    C["blue"],   width=2.0)
                if f"{cty}_HY"    in r_cred.columns:
                    line(fig_cr, r_cred[f"{cty}_HY"].dropna(),    "HY OAS",    C["red"],    width=2.0)
                if f"{cty}_EMSov" in r_cred.columns:
                    line(fig_cr, r_cred[f"{cty}_EMSov"].dropna(), "EM Sov",    C["orange"], width=2.0)
                add_events(fig_cr, start)
                chart(fig_cr, f"credit_{cty}")

                note("Sources: Bloomberg L.P.; ICE BofA indices; JPMorgan EMBIG. "
                     "IG/HY = investment grade / high yield option-adjusted spreads. "
                     "EM Sov = EMBIG stripped spread.")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 6 — Equity & FX
        # ══════════════════════════════════════════════════════════════════════
        if "📊 Equity & FX" in tm:
            with tm["📊 Equity & FX"]:
                r_ef = clip(eqfx, start, end)
                eq_c = f"{cty}_EQ"; fx_c = f"{cty}_FX"; vol_c = f"{cty}_VOL"

                col_ea, col_eb = st.columns([3, 2])
                with col_ea:
                    label("Equity Index  (rebased to 100 at period start, log scale)")
                    fig_eq = base_fig(height=CHART_H)
                    if eq_c in r_ef.columns:
                        s = r_ef[eq_c].dropna()
                        s = s / s.iloc[0] * 100
                        line(fig_eq, s, "Equity", CURRENCY_COLOR.get(cty, C["navy"]))
                    add_events(fig_eq, start)
                    fig_eq.update_yaxes(type="log",
                                        tickvals=[50,75,100,150,200,300],
                                        ticktext=["50","75","100","150","200","300"])
                    chart(fig_eq, f"eq_{cty}")

                with col_eb:
                    label("Implied Volatility")
                    fig_vl = base_fig(height=CHART_H)
                    if vol_c in r_ef.columns:
                        area(fig_vl, r_ef[vol_c].dropna(), "VOL", C["orange"])
                    add_events(fig_vl, start)
                    chart(fig_vl, f"vol_{cty}")

                if fx_c in r_ef.columns:
                    label("FX Index  (trade-weighted, rebased to 100 at period start)")
                    fig_fx = base_fig(height=XS_H)
                    s = r_ef[fx_c].dropna()
                    s = s / s.iloc[0] * 100
                    line(fig_fx, s, "FX", CURRENCY_COLOR.get(cty, C["gray"]))
                    add_events(fig_fx, start)
                    chart(fig_fx, f"fx_{cty}")

                note("Sources: Bloomberg L.P.; BIS. "
                     "Equity and FX rebased to 100 at start of selected sample period.")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 7 — Cross-Currency Basis
        # ══════════════════════════════════════════════════════════════════════
        if "🌐 Cross-Currency" in tm:
            with tm["🌐 Cross-Currency"]:
                r_xccy = clip(xccy, start, end)
                my_cols = [c for c in r_xccy.columns if c.startswith(f"{cty}_Xccy")]

                if not my_cols:
                    st.info("Cross-currency basis data not available for this currency.")
                else:
                    pairs = sorted(set("_".join(c.split("_")[2:4])
                                   for c in my_cols if len(c.split("_")) >= 4))
                    if not pairs:
                        pairs = ["All"]

                    x1, x2, x3, x4 = st.columns(4)
                    tenor_rank = {"3M": 0, "1Y": 1, "5Y": 2, "10Y": 3}
                    ranked_cols = sorted(
                        my_cols,
                        key=lambda c: tenor_rank.get(c.split("_")[-1], 99),
                    )
                    with x1:
                        metric_card("Basis Short", latest_value(r_xccy, ranked_cols[0] if ranked_cols else None), unit=" bps")
                    with x2:
                        metric_card("Basis 1Y", latest_value(r_xccy, next((c for c in my_cols if c.endswith("_1Y")), None)), unit=" bps")
                    with x3:
                        metric_card("Basis 5Y", latest_value(r_xccy, next((c for c in my_cols if c.endswith("_5Y")), None)), unit=" bps")
                    with x4:
                        metric_card("Basis 10Y", latest_value(r_xccy, next((c for c in my_cols if c.endswith("_10Y")), None)), unit=" bps")

                    label("Cross-Currency Basis Swaps vs SOFR  (basis points)")
                    note("Negative = non-USD currency at a discount to SOFR. "
                         + LOG_NOTES.get(cty, {}).get("Xccy", ""))
                    pair_tabs = st.tabs([p.replace("_", " / ") for p in pairs])

                    for pt, pair in zip(pair_tabs, pairs):
                        with pt:
                            pair_cs = [(c, c.split("_")[-1]) for c in my_cols if pair in c]
                            pair_cs.sort(key=lambda x: parse_tenor_label(x[1]))
                            fig_xc = base_fig(height=CHART_H)
                            for k, (col_x, tn_x) in enumerate(pair_cs):
                                if col_x in r_xccy.columns:
                                    line(fig_xc, r_xccy[col_x].dropna(), tn_x,
                                         TENOR_COLORS[k % len(TENOR_COLORS)], width=1.8)
                            hline_zero(fig_xc)
                            add_events(fig_xc, start)
                            chart(fig_xc, f"xccy_{cty}_{pair}")

                    # Term structure snapshot
                    st.divider()
                    label("Cross-Currency Basis Term Structure  (latest date)")
                    fig_xt = xs_fig()
                    for pi, pair in enumerate(pairs):
                        p_cs = sorted([(c, c.split("_")[-1]) for c in my_cols if pair in c],
                                      key=lambda x: parse_tenor_label(x[1]))
                        xv2, yv2 = [], []
                        for col_x, tn_x in p_cs:
                            if col_x in xccy.columns:
                                v = xccy[col_x].dropna()
                                if not v.empty:
                                    yr = parse_tenor_label(tn_x)
                                    if not np.isnan(yr):
                                        xv2.append(yr); yv2.append(float(v.iloc[-1]))
                        if xv2:
                            fig_xt.add_trace(go.Scatter(
                                x=xv2, y=yv2, name=pair.replace("_", " / "),
                                mode="lines+markers",
                                line=dict(color=TENOR_COLORS[pi % len(TENOR_COLORS)], width=2.0),
                                marker=dict(size=5),
                            ))
                    hline_zero(fig_xt)
                    fig_xt.update_xaxes(title_text="Maturity (years)")
                    fig_xt.update_yaxes(title_text="Basis (bps)")
                    chart(fig_xt, f"xccy_term_{cty}")

                    note("Sources: Haver Analytics. "
                         "Cross-currency basis swaps, non-collateralised, fixed reset vs SOFR.")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f'<p style="font-size:0.72rem;color:{C["sub"]}">'
    "<strong>Sources:</strong> Bloomberg L.P.; Haver Analytics; Office of Financial Research (OFR); "
    "DTCC; ICE BofA; JPMorgan; LSEG; BIS. IMF staff calculations. "
    "OIS = overnight indexed swap; GC = general collateral; SC = special collateral; "
    "bps = basis points. Spreads computed as repo minus matched-maturity OIS. "
    "Cross-currency basis: Haver INTDAILY (2023–present).</p>",
    unsafe_allow_html=True,
)

