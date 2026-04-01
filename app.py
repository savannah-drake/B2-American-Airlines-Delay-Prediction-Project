import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AA Flight Deck v3.95", layout="wide", page_icon="✈️")

# --- 1. DATA ENGINE ---
@st.cache_data
def sync_data_engine():
    f_path = 'data/Flight_Data_With_Weather_FINAL.csv'
    if not os.path.exists(f_path): f_path = 'data/Processed_Assignment_Weather_Data.csv'
    df = pd.read_csv(f_path)
    
    # Pre-processing Sync
    df['SPOILED_HRS'] = df['TOTAL_SPOILED_HRS'].fillna(0.0)
    df['IS_COLLAPSE'] = (df['SPOILED_HRS'] >= 6.0).astype(int)
    df['BASE'] = df['BASE'].astype('category')
    df['RIGIDITY'] = df['SEQ_TTL_LEGS'] / (df['TOTAL_BLOCKED_HRS'].fillna(1) + 1)
    df['DHD_LEVERAGE'] = df['IN_SEQ_DHD'].fillna(0) * (df['SEQ_TTL_LEGS'] + 1)
    df['NET_VULN'] = 10 - df['VIS_MILES'].fillna(10.0)
    
    features = ['DHD_LEVERAGE', 'RIGIDITY', 'NET_VULN', 'TOTAL_BLOCKED_HRS', 'SEQ_TTL_LEGS']
    X = df[features + ['BASE']].copy()
    y = df['IS_COLLAPSE']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = HistGradientBoostingClassifier(
        max_iter=1000, learning_rate=0.005, l2_regularization=200.0, random_state=42
    ).fit(X_train, y_train)
    
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return df, model, features, auc

df, model_pre, features, final_auc = sync_data_engine()

# --- INITIALIZE STATE ---
if 'stage' not in st.session_state: st.session_state.stage = 'preemptive'
if 'live_risk' not in st.session_state: st.session_state.live_risk = 0
if 'accum_delay' not in st.session_state: st.session_state.accum_delay = 0
if 'total_stress' not in st.session_state: st.session_state.total_stress = 0
if 'history' not in st.session_state: st.session_state.history = []

# --- LUMINOUS THEME ---
risk_val = st.session_state.live_risk
bg_color = "#0a192f" if risk_val < 15 else "#161b22" if risk_val < 45 else "#0d0202"
accent_color = "#00d4ff"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;700&family=JetBrains+Mono&display=swap');
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {bg_color};
        transition: background-color 1.5s ease;
        font-family: 'Lexend', sans-serif;
    }}
    h1, h2, h3, p, label, .stMarkdown, span {{ color: #ffffff !important; }}
    .stMetric {{ 
        background: rgba(255, 255, 255, 0.08); 
        border: 1px solid rgba(255, 255, 255, 0.3); 
        border-radius: 15px; padding: 15px; 
    }}
    [data-testid="stMetricValue"] {{ color: {accent_color} !important; font-family: 'JetBrains Mono'; font-weight: 700; }}
    .pilot-card {{ 
        background: rgba(255, 255, 255, 0.05); 
        border: 2px solid {accent_color}; 
        padding: 20px; border-radius: 15px; margin-top: 10px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }}
    </style>
    """, unsafe_allow_html=True)

# --- PHASE 1: PREEMPTIVE ---
if st.session_state.stage == 'preemptive':
    st.title("✈️ DISPATCH CONTROL CENTER")
    st.markdown(f"<h4 style='color:{accent_color}; font-family:JetBrains Mono'>SYSTEM RELIABILITY (AUC): {final_auc:.4f}</h4>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        u_base = st.selectbox("OPERATIONAL HUB", sorted(df['BASE'].cat.categories))
        u_legs = st.slider("TOTAL SEQUENCE LEGS", 1, 15, 4)
        u_block = st.slider("TOTAL BLOCK HOURS", 1.0, 30.0, 12.0)

    with c2:
        u_vis = st.slider("HUB VISIBILITY (MILES)", 0.0, 10.0, 10.0)
        u_dhd = st.number_input("CREW DEADHEADS (DHD)", 0, 10, 0)

        # --- UPDATED HISTORICAL LOOKUP (BLOCK HOUR AWARE) ---
        # Filters by Hub, exact Legs, Vis window, AND Block Hour window
        m = df[
            (df['BASE'] == u_base) & 
            (df['SEQ_TTL_LEGS'] == u_legs) & 
            (df['VIS_MILES'].between(u_vis-2, u_vis+2)) &
            (df['TOTAL_BLOCKED_HRS'].between(u_block-2, u_block+2)) # THE FIX
        ]
        
        avg_s = m['SPOILED_HRS'].mean() if not m.empty else 0
        worst_s = m['SPOILED_HRS'].max() if not m.empty else 0
        
        st.markdown(f"""
        <div class='pilot-card'>
            <h3 style='margin-top:0; color:{accent_color}'>📊 HISTORICAL GROUND TRUTH</h3>
            <p>Similar Profiles Found: <b>{len(m)}</b></p>
            <p>Avg. Historical Spoilage: <b>{avg_s:.2f}h</b></p>
            <p style='color:#ff4b4b !important'>Worst Reported Spoilage: <b>{worst_s:.2f}h</b></p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("RUN PREEMPTIVE RISK ASSESSMENT", type="primary"):
        test_case = pd.DataFrame({
            'DHD_LEVERAGE': [u_dhd * (u_legs + 1)], 'RIGIDITY': [u_legs / (u_block + 1)],
            'NET_VULN': [10 - u_vis], 'TOTAL_BLOCKED_HRS': [u_block], 'SEQ_TTL_LEGS': [u_legs], 'BASE': [u_base]
        })
        test_case['BASE'] = pd.Categorical(test_case['BASE'], categories=df['BASE'].cat.categories)
        p_risk = model_pre.predict_proba(test_case)[:, 1][0] * 100
        
        st.session_state.pre_risk = p_risk
        st.session_state.live_risk = p_risk
        st.session_state.pre_data = {'u_legs': u_legs, 'u_base': u_base, 'avg_spoil': avg_s}
        st.session_state.history = [{'leg': 0, 'risk': p_risk, 'delay': 0}]
        st.session_state.stage = 'dynamic'
        st.rerun()

# --- PHASE 2: DYNAMIC ---
elif st.session_state.stage == 'dynamic':
    d = st.session_state.pre_data
    curr_leg = len(st.session_state.history)
    
    st.title(f"🎮 LIVE OPS: {d['u_base']} HUB")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("RECALCULATED RISK", f"{st.session_state.live_risk:.1f}%")
    m2.metric("EXPECTED DELAY", f"{st.session_state.accum_delay}m")
    m3.metric("STRESS PENALTY", f"+{st.session_state.total_stress:.2f}%")

    if curr_leg <= d['u_legs']:
        with st.sidebar:
            st.header(f"📍 Landing: Leg {curr_leg}")
            l_vis = st.slider("Destination Visibility", 0.0, 10.0, 10.0)
            l_delay = st.number_input("Minutes Delay Added", 0, 180, 0)
            
            st.markdown("---")
            speed_up = st.checkbox("⚡ Apply Speed-Up?")
            recovery = st.slider("Minutes Recovered", 0, 30, 15) if speed_up else 0

            if st.button("UPDATE SYSTEM"):
                # Net Change = Delay - Recovery
                net_change = l_delay - recovery
                st.session_state.accum_delay = max(0, st.session_state.accum_delay + net_change)
                
                # Stress math (0.05% per minute recovered)
                if speed_up:
                    st.session_state.total_stress += (recovery * 0.05)
                
                # Risk Formula: Base + ((Delay * Impact) * VisMult) + Stress
                vis_mult = 1.0 + (10 - l_vis) * 0.05 
                impact = (st.session_state.accum_delay * (0.35 + (d['u_legs'] * 0.05))) * vis_mult
                
                st.session_state.live_risk = min(99.9, st.session_state.pre_risk + impact + st.session_state.total_stress)
                
                st.session_state.history.append({'leg': curr_leg, 'risk': st.session_state.live_risk, 'delay': st.session_state.accum_delay})
                st.rerun()

    h_df = pd.DataFrame(st.session_state.history)
    fig = go.Figure(go.Scatter(x=h_df['leg'], y=h_df['risk'], mode='lines+markers', line=dict(color=accent_color, width=6), fill='tozeroy'))
    fig.update_layout(template="plotly_dark", yaxis_range=[0, 105], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    if st.button("TERMINATE OPS"):
        for k in ['stage', 'live_risk', 'accum_delay', 'total_stress', 'history']:
            if k in st.session_state: del st.session_state[k]
        st.rerun()