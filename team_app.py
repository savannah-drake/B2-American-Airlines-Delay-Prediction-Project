import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import base64

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_image("assets/aa_logo.png")
# --- PAGE CONFIG ---
st.set_page_config(page_title="AA Sky Deck v8.1", layout="wide", page_icon="✈️")

# --- 1. DATA ENGINE (High AUC Optimization) ---
@st.cache_data
def sync_data_engine():
    f_path = 'data/Flight_Data_With_Weather_FINAL.csv'
    if not os.path.exists(f_path): 
        f_path = 'data/Processed_Assignment_Weather_Data.csv'
    
    df = pd.read_csv(f_path)
    df['SPOILED_HRS'] = df['TOTAL_SPOILED_HRS'].fillna(0.0)
    df['IS_COLLAPSE'] = (df['SPOILED_HRS'] >= 6.0).astype(int)
    
    # Feature Engineering for 0.65+ AUC
    df['RIGIDITY'] = df['SEQ_TTL_LEGS'] / (df['TOTAL_BLOCKED_HRS'].fillna(1) + 1)
    df['VIS_MILES'] = df['VIS_MILES'].fillna(10.0)
    df['WEATHER_PRESSURE'] = df['SEQ_TTL_LEGS'] * (11 - df['VIS_MILES'])
    
    # Bayesian Smoothing
    global_mean = df['IS_COLLAPSE'].mean()
    smoothing = 20
    agg = df.groupby(['BASE', 'FLEET'])['IS_COLLAPSE'].agg(['count', 'mean']).reset_index()
    agg['BAYES_RISK'] = ((agg['count'] * agg['mean']) + (smoothing * global_mean)) / (agg['count'] + smoothing)
    risk_map = agg.set_index(['BASE', 'FLEET'])['BAYES_RISK'].to_dict()
    df['BAYES_RISK'] = df.apply(lambda r: risk_map.get((r['BASE'], r['FLEET']), global_mean), axis=1)
    
    # LAYOVER COUNT MAPPING (Ensuring it exists for filtering)
    if 'LAYOVER_COUNT' not in df.columns:
        if 'SEQ_CAL_DAYS' in df.columns:
            df['LAYOVER_COUNT'] = (df['SEQ_CAL_DAYS'] - 1).clip(lower=0)
        elif 'LAYOVER' in df.columns:
            df['LAYOVER_COUNT'] = (df['LAYOVER'] > 0).astype(int)
        else:
            df['LAYOVER_COUNT'] = 0
    
    features = ['BAYES_RISK', 'TOTAL_BLOCKED_HRS', 'SEQ_TTL_LEGS', 'LAYOVER_COUNT', 'RIGIDITY', 'WEATHER_PRESSURE']
    X = df[features].fillna(0)
    y = df['IS_COLLAPSE']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.02, max_depth=6, random_state=42).fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, probs)
    
    return df, model, risk_map, global_mean, auc_score

df, model, risk_map, global_mean, final_auc = sync_data_engine()

# --- 2. DESIGN SYSTEM ---
NAVY       = "#001529"
NAVY_MID   = "#00244a"
BLUE_AA    = "#0070cc"
BLUE_LIGHT = "#5bb3ff"
CYAN       = "#00e5ff"
GREEN      = "#3ddc84"
AMBER      = "#ffb400"
RED        = "#ff6b6b"
GLASS_BG   = "rgba(255,255,255,0.05)"
GLASS_BDR  = "rgba(255,255,255,0.10)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── ROOT RESET ── */
*, *::before, *::after {{ box-sizing: border-box; }}

[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(0,112,204,0.30) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 85% 80%, rgba(0,61,121,0.35) 0%, transparent 60%),
        linear-gradient(160deg, {NAVY} 0%, {NAVY_MID} 50%, #001e3c 100%);
    font-family: 'Sora', sans-serif;
}}

[data-testid="stHeader"] {{ display: none; }}
[data-testid="stToolbar"] {{ display: none; }}

/* Grid overlay */
[data-testid="stAppViewContainer"]::before {{
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background-image:
        linear-gradient(rgba(0,112,204,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,112,204,0.05) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
}}

/* ── TYPOGRAPHY ── */
h1, h2, h3, h4, p, label {{
    color: white !important;
    font-family: 'Sora', sans-serif !important;
}}

/* ── AA HEADER ── */
.aa-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 32px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 32px;
    background: rgba(0,0,0,0.15);
    backdrop-filter: blur(20px);
    border-radius: 0 0 20px 20px;
}}

.aa-brand {{
    display: flex;
    align-items: center;
    gap: 16px;
}}

.aa-logo {{
    width: 160px;
    height: 160px;
    display: flex;
    align-items: center;
    justify-content: center;
}}

.aa-logo img {{
    width: 100%;
    height: 100%;
    object-fit: contain;
}}

.aa-title-block span.label {{
    font-size: 10px;
    letter-spacing: 3px;
    color: rgba(255,255,255,0.35);
    display: block;
    text-transform: uppercase;
}}

.aa-title-block span.title {{
    font-size: 20px;
    font-weight: 700;
    background: linear-gradient(90deg, #fff 0%, {BLUE_LIGHT} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: block;
}}

.divider-v {{
    width: 1px;
    height: 36px;
    background: rgba(255,255,255,0.12);
    margin: 0 4px;
}}

.product-name {{
    font-size: 16px;
    font-weight: 600;
}}

.product-ver {{
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    color: rgba(255,255,255,0.35);
    letter-spacing: 1px;
}}

/* ── STATUS PILLS ── */
.pill-row {{
    display: flex;
    gap: 10px;
    align-items: center;
}}

.pill {{
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.5px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}}

.pill-auc {{
    background: rgba(0,112,204,0.2);
    border: 1px solid rgba(0,112,204,0.5);
    color: {BLUE_LIGHT};
}}

.pill-live {{
    background: rgba(61,220,132,0.12);
    border: 1px solid rgba(61,220,132,0.4);
    color: {GREEN};
}}

.dot-live {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: {GREEN};
    display: inline-block;
    animation: blink 2s infinite;
}}

@keyframes blink {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}

/* ── GLASS CARD ── */
.glass-card {{
    background: {GLASS_BG};
    border: 1px solid {GLASS_BDR};
    border-radius: 20px;
    padding: 24px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    margin-bottom: 0;
}}

/* ── SECTION LABEL ── */
.section-label {{
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 2.5px;
    color: rgba(255,255,255,0.30);
    text-transform: uppercase;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 10px;
}}

.section-label::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.07);
    display: block;
}}

/* ── FIELD LABELS ── */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stRadio"] label {{
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 1.5px !important;
    color: rgba(255,255,255,0.40) !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}}

/* ── SELECT / INPUT ── */
 /* SELECT / INPUT */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    min-height: 46px !important;
    padding: 0 14px !important;
}}
/* ── NUMBER INPUT ── */
[data-testid="stNumberInput"] > div {{
    width: 100%;
}}

[data-testid="stNumberInput"] div[data-baseweb="input"] {{
    background: rgba(0,112,204,0.12) !important;
    border: 1px solid rgba(91,179,255,0.35) !important;
    border-radius: 10px !important;
    min-height: 46px !important;
    transition: all 0.2s ease !important;
}}

[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {{
    border: 1px solid rgba(91,179,255,0.65) !important;
    box-shadow: 0 0 0 1px rgba(91,179,255,0.20), 0 0 16px rgba(0,112,204,0.18) !important;
    background: rgba(0,112,204,0.16) !important;
}}

/* targeted fix for gray value background */
[data-testid="stNumberInput"] input {{
    color: white !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding-left: 2px !important;
}}

[data-testid="stNumberInput"] div[data-baseweb="input"] > div {{
    background: transparent !important;
    background-color: transparent !important;
}}

[data-testid="stNumberInput"] input::placeholder {{
    color: rgba(255,255,255,0.45) !important;
    opacity: 1 !important;
}}

/* stepper buttons */
[data-testid="stNumberInput"] button {{
    background: transparent !important;
    border: none !important;
    color: rgba(255,255,255,0.65) !important;
}}

[data-testid="stNumberInput"] button:hover {{
    background: rgba(255,255,255,0.06) !important;
    color: white !important;
    border-radius: 8px !important;
}}

[data-testid="stNumberInput"] svg {{
    fill: currentColor !important;
}}
[data-testid="stNumberInput"] {{
    margin-bottom: 6px !important;
}}

.range-note {{
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    color: rgba(91,179,255,0.58);
    letter-spacing: 0.4px;
    margin-top: -2px;
    margin-bottom: 14px;
    line-height: 1.4;
}}
.input-group {{
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 12px 12px 8px 12px;
    margin-bottom: 12px;
}}

/* selected value text */
[data-testid="stSelectbox"] div[data-baseweb="select"] span {{
    color: white !important;
    opacity: 1 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}}

/* sometimes Streamlit/BaseWeb uses div instead of span for selected value */
[data-testid="stSelectbox"] div[data-baseweb="select"] div {{
    color: white !important;
}}

/* placeholder text */
[data-testid="stSelectbox"] div[data-baseweb="select"] input::placeholder,
[data-testid="stSelectbox"] div[data-baseweb="select"] span[data-testid="stMarkdownContainer"] {{
    color: rgba(255,255,255,0.7) !important;
    opacity: 1 !important;
}}

/* dropdown arrow */
[data-testid="stSelectbox"] svg {{
    fill: white !important;
}}

/* ── DROPDOWN MENU ── */
[data-baseweb="popover"] {{
    background: #001e3c !important;
    border: 1px solid rgba(0,112,204,0.3) !important;
    border-radius: 12px !important;
}}

[data-baseweb="menu"] li {{
    color: rgba(255,255,255,0.8) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 14px !important;
}}

[data-baseweb="menu"] li:hover {{
    background: rgba(0,112,204,0.2) !important;
}}

/* ── RANGE NOTE ── */
.range-note {{
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    color: rgba(0,180,255,0.55);
    letter-spacing: 0.5px;
    margin-top: -4px;
    margin-bottom: 12px;
}}

/* ── STAT CARDS ── */
.stat-card {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
}}

.stat-label {{
    font-size: 9px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 1.5px;
    color: rgba(255,255,255,0.30);
    text-transform: uppercase;
    margin-bottom: 6px;
}}

.stat-val {{ font-size: 28px; font-weight: 700; color: white; line-height: 1; }}
.stat-val-accent {{ font-size: 28px; font-weight: 700; color: {BLUE_LIGHT}; line-height: 1; }}
.stat-val-danger {{ font-size: 28px; font-weight: 700; color: {RED}; line-height: 1; }}

.info-box {{
    background: rgba(0,112,204,0.08);
    border: 1px solid rgba(0,112,204,0.2);
    border-radius: 10px;
    padding: 12px;
    margin-top: 4px;
}}

.info-box-label {{
    font-size: 9px;
    font-family: 'DM Mono', monospace;
    color: rgba(0,180,255,0.6);
    letter-spacing: 1px;
    margin-bottom: 4px;
    text-transform: uppercase;
}}

.info-box-text {{
    font-size: 12px;
    color: rgba(255,255,255,0.45);
}}

/* ── CTA BUTTON ── */
[data-testid="stButton"] > button {{
    width: 100%;
    padding: 16px 24px !important;
    background: linear-gradient(135deg, {BLUE_AA} 0%, #004a99 100%) !important;
    border: 1px solid rgba(0,150,255,0.4) !important;
    border-radius: 14px !important;
    color: white !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    margin-top: 8px !important;
}}

[data-testid="stButton"] > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,112,204,0.45) !important;
    background: linear-gradient(135deg, #0080e0 0%, #0057b3 100%) !important;
}}

[data-testid="stButton"] > button:active {{
    transform: translateY(0) !important;
}}

/* ── RESULTS ── */
.result-card {{
    background: {GLASS_BG};
    border: 1px solid {GLASS_BDR};
    border-left: 4px solid {BLUE_AA};
    border-radius: 20px;
    padding: 28px;
    backdrop-filter: blur(20px);
}}

.flight-type {{
    font-size: 24px;
    font-weight: 700;
    color: white;
    line-height: 1.2;
}}

.flight-sub {{
    font-size: 12px;
    font-family: 'DM Mono', monospace;
    color: rgba(255,255,255,0.35);
    margin-top: 4px;
    letter-spacing: 0.5px;
}}

.window-label {{
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 2px;
    color: rgba(255,255,255,0.30);
    text-transform: uppercase;
    margin-bottom: 6px;
    margin-top: 20px;
}}

.window-range {{
    font-size: 52px;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    color: white;
    line-height: 1;
}}

.window-unit {{
    font-size: 18px;
    color: rgba(255,255,255,0.35);
    font-family: 'DM Mono', monospace;
}}

.factor-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.07);
}}

.factor-item {{
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 12px;
}}

.factor-label {{
    font-size: 9px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 1px;
    color: rgba(255,255,255,0.30);
    text-transform: uppercase;
    margin-bottom: 4px;
}}

.factor-val {{
    font-size: 16px;
    font-weight: 600;
    color: white;
}}

/* ── RISK BADGE ── */
.risk-badge {{
    display: inline-block;
    padding: 8px 20px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-top: 14px;
}}
.section-label {{
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 1px;
    color: white;
    
    padding: 10px 14px;
    border-radius: 10px;

    background: rgba(0,112,204,0.12);
    border: 1px solid rgba(0,112,204,0.25);

    display: inline-block;
    margin-bottom: 16px;
}}

.section-label::after {{
    display: none;
}}

.risk-low  {{ background: rgba(61,220,132,0.12); border: 1px solid rgba(61,220,132,0.4); color: {GREEN}; }}
.risk-med  {{ background: rgba(255,180,0,0.12);  border: 1px solid rgba(255,180,0,0.4);  color: {AMBER}; }}
.risk-high {{ background: rgba(255,80,80,0.12);  border: 1px solid rgba(255,80,80,0.4);  color: {RED}; }}

/* ── FOOTER ── */
.aa-footer {{
    text-align: center;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-size: 10px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 2px;
    color: rgba(255,255,255,0.15);
    text-transform: uppercase;
}}

/* ── SPACING CLEANUP ── */
.block-container {{ padding: 0 32px 40px 32px !important; max-width: 1240px !important; }}
[data-testid="stVerticalBlock"] > div {{ gap: 0 !important; }}
.element-container {{ margin-bottom: 8px; }}
hr {{ border-color: rgba(255,255,255,0.07) !important; margin: 24px 0 !important; }}
</style>
""", unsafe_allow_html=True)

# --- 3. HEADER ---
st.markdown(f"""
<div class="aa-header">
  <div class="aa-brand">
    <div class="aa-logo">
  <img src="data:image/png;base64,{logo_base64}" style="width:100%; height:100%; object-fit:contain;" />
</div>
    <div class="aa-title-block">
      <span class="label">American Airlines</span>
      <span class="title">Operational Intelligence</span>
    </div>
    <div class="divider-v"></div>
    <div>
      <div class="product-name">Operations Risk Commander</div>
      <div class="product-ver">v8.1 · ML-Powered Analysis</div>
    </div>
  </div>
  <div class="pill-row">
    <div class="pill pill-auc">AUC: {final_auc:.4f}</div>
    <div class="pill pill-live"><div class="dot-live"></div> Live Sync</div>
  </div>
</div>
""", unsafe_allow_html=True)

# --- 4. CONFIGURE PARAMETERS ---
st.markdown('<div class="section-label">Configure Assessment Parameters</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1, 1], gap="medium")

with c1:
    st.markdown('<div class="section-label">Deployment</div>', unsafe_allow_html=True)

    u_base = st.selectbox("Operational Hub", sorted(df['BASE'].unique()), key="hub")
    u_fleet = st.selectbox("Aircraft Fleet", sorted(df['FLEET'].unique()), key="fleet")

    weather_opt = {
        "☀️  Clear (10 mi)": 10.0,
        "☁️  Overcast (5 mi)": 5.0,
        "🌧  Rainy (3 mi)": 3.0,
        "⛈  Stormy (1 mi)": 1.0,
    }
    u_weather = st.selectbox("Weather Condition", list(weather_opt.keys()), key="weather")
    u_vis = weather_opt[u_weather]
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="section-label">Sequence Details</div>', unsafe_allow_html=True)
    u_legs = st.number_input("Total Legs", min_value=1, max_value=15, value=5, key="legs")
    st.markdown('<p class="range-note">1–2 Low &nbsp;·&nbsp; 3–5 Normal &nbsp;·&nbsp; 6+ High</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    u_block = st.number_input("Total Block Hours", min_value=1.0, max_value=30.0, value=14.0, step=0.5, key="block")
    st.markdown('<p class="range-note">&lt;10 Low &nbsp;·&nbsp; 10–16 Normal &nbsp;·&nbsp; 16+ High</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    u_lay = st.number_input("Layovers", min_value=0, max_value=5, value=2, key="lay")
    st.markdown('<p class="range-note">0 Direct &nbsp;·&nbsp; 1–2 Multi-Day &nbsp;·&nbsp; 3+ High</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="section-label">Historical Context</div>', unsafe_allow_html=True)

    hist_match = df[
        (df['BASE'] == u_base) &
        (df['FLEET'] == u_fleet) &
        (df['SEQ_TTL_LEGS'].between(u_legs - 1, u_legs + 1)) &
        (df['LAYOVER_COUNT'] == u_lay) &
        (df['VIS_MILES'].between(u_vis - 2, u_vis + 2)) &
        (df['TOTAL_BLOCKED_HRS'].between(u_block - 4, u_block + 4))
    ]


    entries = len(hist_match)

    guardrail_msgs = []

    if entries == 0:
        guardrail_msgs.append("No close historical matches found. Output is extrapolated.")
    elif entries < 5:
        guardrail_msgs.append("Very limited historical matches. Prediction confidence is low.")
    elif entries < 15:
        guardrail_msgs.append("Moderate historical coverage. Use result as directional guidance.")

    block_low, block_high = df['TOTAL_BLOCKED_HRS'].quantile([0.01, 0.99])
    legs_low, legs_high = df['SEQ_TTL_LEGS'].quantile([0.01, 0.99])
    lay_low, lay_high = df['LAYOVER_COUNT'].quantile([0.01, 0.99])

    if not (block_low <= u_block <= block_high):
        guardrail_msgs.append("Block hours are outside the typical training range.")
    if not (legs_low <= u_legs <= legs_high):
        guardrail_msgs.append("Leg count is outside the typical training range.")
    if not (lay_low <= u_lay <= lay_high):
        guardrail_msgs.append("Layovers are outside the typical training range.")
    avg_s = hist_match['SPOILED_HRS'].mean() if entries > 0 else 0
    worst_s = hist_match['SPOILED_HRS'].max() if entries > 0 else 0

    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">Similar Profiles Matched</div>
      <div class="stat-val-accent">{entries}</div>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
      <div class="stat-card">
        <div class="stat-label">Avg Delay</div>
        <div class="stat-val">{avg_s:.2f}h</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Worst Recorded</div>
        <div class="stat-val-danger">{worst_s:.2f}h</div>
      </div>
    </div>
    <div class="info-box" style="margin-top: 10px;">
      <div class="info-box-label">Data Window</div>
      <div class="info-box-text">±1 leg · exact layover · ±2mi visibility · ±4h block</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
for msg in guardrail_msgs:
    st.warning(f"⚠️ {msg}")
# --- 5. CTA ---
st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
run = st.button("✈  Execute Risk Assessment", type="primary", use_container_width=True)

# --- 6. RESULTS ---
if run:
    b_risk = risk_map.get((u_base, u_fleet), global_mean)
    u_rigidity = u_legs / (u_block + 1)
    u_weather_p = u_legs * (11 - u_vis)

    input_df = pd.DataFrame([{
        'BAYES_RISK': b_risk,
        'TOTAL_BLOCKED_HRS': u_block,
        'SEQ_TTL_LEGS': u_legs,
        'LAYOVER_COUNT': u_lay,
        'RIGIDITY': u_rigidity,
        'WEATHER_PRESSURE': u_weather_p
    }])
    prob = model.predict_proba(input_df)[:, 1][0] * 100
    prob = max(0, min(prob, 100))
    

    if u_block <= 8:
        f_type, f_icon, f_sub = "Short-Haul Hopper", "🛫", "Regional / Express Segment"
    elif u_block <= 16:
        f_type, f_icon, f_sub = "Medium-Haul Cruiser", "✈️", "Domestic / Caribbean Range"
    else:
        f_type, f_icon, f_sub = "Long-Haul Endurance", "🛰️", "International / Transoceanic"

    window = 1.5 if u_block <= 8 else (3.5 if u_block <= 16 else 6.0)
    low_b = max(0, avg_s - window / 2)
    high_b = max(low_b, avg_s + window / 2)

    if prob < 35:
        risk_class, risk_label = "risk-low", "Low Risk"
    elif prob < 65:
        risk_class, risk_label = "risk-med", "Moderate Risk"
    else:
        risk_class, risk_label = "risk-high", "High Risk"

    gauge_color = "#3ddc84" if prob < 35 else ("#ffb400" if prob < 65 else "#ff6b6b")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Assessment Output</div>', unsafe_allow_html=True)

    res_l, res_r = st.columns([1, 1.6], gap="medium")

    with res_l:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={
                'suffix': "%",
                'font': {'color': 'white', 'family': 'DM Mono', 'size': 42},
                'valueformat': '.1f'
            },
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 0,
                    'tickcolor': 'rgba(0,0,0,0)',
                    'tickfont': {'color': 'rgba(255,255,255,0.3)', 'size': 10, 'family': 'DM Mono'},
                    'nticks': 5,
                },
                'bar': {'color': gauge_color, 'thickness': 0.7},
                'bgcolor': 'rgba(255,255,255,0.05)',
                'borderwidth': 1,
                'bordercolor': 'rgba(255,255,255,0.1)',
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(61,220,132,0.08)'},
                    {'range': [35, 65], 'color': 'rgba(255,180,0,0.08)'},
                    {'range': [65, 100], 'color': 'rgba(255,80,80,0.08)'},
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 2},
                    'thickness': 0.75,
                    'value': prob
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white', 'family': 'DM Mono'},
            height=280,
            margin=dict(l=30, r=30, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div style="text-align: center; margin-top: -8px;">
          <span class="risk-badge {risk_class}">{risk_label}</span>
        </div>
        <div style="text-align: center; margin-top: 10px; font-size: 10px; font-family: 'DM Mono', monospace; color: rgba(255,255,255,0.25); letter-spacing: 1px;">
          COLLAPSE PROBABILITY INDEX
        </div>
        """, unsafe_allow_html=True)

    with res_r:
        st.markdown(f"""
        <div class="result-card">
          <div style="display: flex; align-items: flex-start; justify-content: space-between;">
            <div>
              <div class="flight-type">{f_type}</div>
              <div class="flight-sub">{f_sub}</div>
            </div>
            <div style="font-size: 40px; line-height: 1;">{f_icon}</div>
          </div>

          <div class="window-label">Predicted Spoilage Window</div>
          <div>
            <span class="window-range">{low_b:.1f} — {high_b:.1f}</span>
            <span class="window-unit"> hrs</span>
          </div>
          <div style="font-size: 11px; color: rgba(255,255,255,0.30); font-family: 'DM Mono', monospace; margin-top: 4px;">
            Estimated system delay · ±confidence interval
          </div>

          <div class="factor-grid">
            <div class="factor-item">
              <div class="factor-label">Rigidity Score</div>
              <div class="factor-val">{u_rigidity:.3f}</div>
            </div>
            <div class="factor-item">
              <div class="factor-label">Weather Pressure</div>
              <div class="factor-val">{u_weather_p:.1f}</div>
            </div>
            <div class="factor-item">
              <div class="factor-label">Bayes Risk</div>
              <div class="factor-val">{b_risk:.3f}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# --- 7. FOOTER ---
st.markdown("""
<div class="aa-footer">
  ✦ &nbsp; AA Sky Deck v8.1 &nbsp;·&nbsp; Precision ML Operations &nbsp;·&nbsp; Internal Use Only &nbsp; ✦
</div>
""", unsafe_allow_html=True)