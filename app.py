import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tamil Nadu Crop Advisor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f2eb; }

.hero {
    background: #1a3a2a; border-radius: 16px;
    padding: 36px 40px 28px; margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.hero::before {
    content: '🌾'; font-size: 120px; position: absolute;
    right: 40px; top: 10px; opacity: 0.12;
}
.hero h1 {
    font-family: 'DM Serif Display', serif; color: #e8f5e0;
    font-size: 2.2rem; margin: 0 0 6px 0; font-weight: 400;
}
.hero p { color: #94b88a; margin: 0; font-size: 1rem; }

.input-card {
    background: white; border-radius: 14px; padding: 24px;
    margin-bottom: 24px; border: 1px solid #e8e2d8;
}
.crop-card {
    background: white; border-radius: 14px; padding: 22px 24px;
    margin-bottom: 14px; border: 1px solid #e8e2d8;
    border-left: 5px solid #2d7a4f;
}
.crop-card.moderate { border-left-color: #e0a020; }
.crop-card.poor { border-left-color: #cc4444; }
.crop-name {
    font-family: 'DM Serif Display', serif; font-size: 1.4rem;
    color: #1a3a2a; margin: 0 0 4px 0;
}
.score-badge {
    display: inline-block; background: #e8f5e0; color: #2d7a4f;
    font-weight: 600; font-size: 0.85rem; padding: 3px 10px;
    border-radius: 20px; margin-bottom: 14px;
}
.score-badge.moderate { background: #fff3d4; color: #c47f00; }
.score-badge.poor { background: #fde8e8; color: #b33333; }
.metric-row { display: flex; gap: 10px; flex-wrap: wrap; }
.metric-chip {
    background: #f5f2eb; border-radius: 8px; padding: 6px 12px;
    font-size: 0.82rem; color: #3d5a45; border: 1px solid #e0dbd0;
}
.insight-box {
    background: #f0f8f4; border-radius: 10px; padding: 12px 14px;
    border: 1px solid #c8e6d4; font-size: 0.85rem; color: #2d5a3d;
    margin-top: 12px; line-height: 1.6;
}
.avoid-card {
    background: #fff8f8; border-radius: 12px; padding: 14px 18px;
    margin-bottom: 10px; border: 1px solid #f0d8d8;
    display: flex; align-items: center; gap: 10px;
}
.avoid-crop { font-weight: 600; color: #8b2020; font-size: 1rem; }
.alt-card {
    background: #f0f8f4; border-radius: 12px; padding: 12px 16px;
    margin-bottom: 8px; border: 1px solid #c8e6d4;
    display: flex; justify-content: space-between; align-items: center;
}
.alt-name { font-weight: 600; color: #1a5c35; }
.alt-score {
    font-size: 0.82rem; color: #2d8a55; background: #e0f5ea;
    padding: 2px 9px; border-radius: 12px;
}
.risk-panel {
    background: white; border-radius: 14px; padding: 22px 24px;
    border: 1px solid #e8e2d8; margin-bottom: 16px;
}
.risk-title {
    font-family: 'DM Serif Display', serif; font-size: 1.1rem;
    color: #1a3a2a; margin: 0 0 14px 0;
}
.risk-safe {
    background: #e8f5e0; color: #1a5c35; border-radius: 10px;
    padding: 12px 16px; font-weight: 600; text-align: center; font-size: 1rem;
}
.risk-moderate {
    background: #fff3d4; color: #9a6200; border-radius: 10px;
    padding: 12px 16px; font-weight: 600; text-align: center; font-size: 1rem;
}
.risk-high {
    background: #fde8e8; color: #8b1a1a; border-radius: 10px;
    padding: 12px 16px; font-weight: 600; text-align: center; font-size: 1rem;
}
.section-label {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #7a8f82;
    margin-bottom: 12px; margin-top: 4px;
}
.no-data {
    background: #f9f7f3; border-radius: 10px; padding: 16px;
    text-align: center; color: #8a9a8f; font-size: 0.92rem;
    border: 1px dashed #d0ccc4;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("FINAL_READY_DATASET.csv")
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df

@st.cache_resource
def load_model():
    return joblib.load("clf_model.pkl")

df = load_data()
clf = load_model()

FEATURES = [
    "Water_Score", "Temp_Score", "Soil_Score",
    "Avg_Rainfall_mm", "Production_MT",
    "District_Temp_Min", "District_Temp_Max",
    "Water_Min", "Water_Max"
]

# ── HELPERS ─────────────────────────────────────────────────────────
def predict_score(row_df):
    proba = clf.predict_proba(row_df[FEATURES])[0]
    return proba[0] * 0.0 + proba[1] * 0.5 + proba[2] * 1.0

def match_label(score):
    if score >= 0.7:   return "✅ Good"
    elif score >= 0.4: return "🟡 Moderate"
    else:              return "❌ Low"

def score_class(score):
    if score >= 0.65:   return ""
    elif score >= 0.45: return "moderate"
    else:               return "poor"

# ── HERO ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Tamil Nadu Crop Advisor</h1>
  <p>AI-powered recommendations based on district rainfall, soil, and historical yield data</p>
</div>
""", unsafe_allow_html=True)

# ── INPUTS ──────────────────────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    district = st.selectbox("📍 District", sorted(df["District"].unique()))
with c2:
    season = st.selectbox("🌦 Season", ["Kharif", "Rabi"])
with c3:
    crop_input = st.selectbox("🔍 Check a specific crop", ["None"] + sorted(df["Crop"].unique()))
st.markdown('</div>', unsafe_allow_html=True)

# ── FILTER & SCORE ──────────────────────────────────────────────────
df_d = df[
    (df["District"] == district) &
    ((df["Season"] == season) | (df["Season"] == "Annual"))
].copy()

if not df_d.empty:
    df_d["ML_Score"] = df_d.apply(
        lambda r: predict_score(pd.DataFrame([r])), axis=1
    )
    df_d["Final_Score"] = 0.6 * df_d["Suitability_Score"] + 0.4 * df_d["ML_Score"]

df_sorted = df_d.sort_values("Final_Score", ascending=False)

top = df_sorted[df_sorted["Final_Score"] > 0.55].head(3)
if top.empty:
    top = df_sorted.head(2)

avoid = df_sorted[df_sorted["Final_Score"] < 0.45]
avoid = avoid[~avoid["Crop"].isin(top["Crop"])].head(3)

existing_crops = set(df_d["Crop"])
alternate = df[
    (~df["Crop"].isin(existing_crops)) &
    (df["Suitability_Score"] > 0.6)
].drop_duplicates(subset=["Crop"]).head(4)

# ── LAYOUT ──────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.markdown('<p class="section-label">Recommended Crops</p>', unsafe_allow_html=True)

    if df_d.empty:
        st.markdown('<div class="no-data">No crop data for this district and season.</div>', unsafe_allow_html=True)
    elif top.empty:
        st.markdown('<div class="no-data">No strongly suitable crops found. Try a different season.</div>', unsafe_allow_html=True)
    else:
        for _, row in top.iterrows():
            sc = score_class(row["Final_Score"])
            water_pct = int(row["Water_Score"] * 100)
            soil_txt = "good soil match" if row["Soil_Score"] >= 0.8 else "partial soil match"
            insight = f"Water availability at {water_pct}% of crop need · {soil_txt} · {row['Season']} crop"

            st.markdown(f"""
            <div class="crop-card {sc}">
              <p class="crop-name">{row['Crop'].title()}</p>
              <span class="score-badge {sc}">Score: {row['Final_Score']:.2f}</span>
              <div class="metric-row">
                <div class="metric-chip">💧 Water {match_label(row['Water_Score'])}</div>
                <div class="metric-chip">🌡 Temp {match_label(row['Temp_Score'])}</div>
                <div class="metric-chip">🌱 Soil {match_label(row['Soil_Score'])}</div>
                <div class="metric-chip">📦 Yield {match_label(row['Productivity_Score'])}</div>
              </div>
              <div class="insight-box">{insight}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<p class="section-label" style="margin-top:24px">Alternate Options</p>', unsafe_allow_html=True)
    if alternate.empty:
        st.markdown('<div class="no-data">No alternate crops found.</div>', unsafe_allow_html=True)
    else:
        for _, row in alternate.iterrows():
            st.markdown(f"""
            <div class="alt-card">
              <span class="alt-name">🌿 {row['Crop'].title()}</span>
              <span class="alt-score">Suitability {row['Suitability_Score']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

with right:
    st.markdown('<p class="section-label">Crops to Avoid</p>', unsafe_allow_html=True)
    if avoid.empty:
        st.markdown('<div class="no-data">✅ No high-risk crops this season.</div>', unsafe_allow_html=True)
    else:
        for _, row in avoid.iterrows():
            reason = "insufficient water" if row["Water_Score"] < 0.4 else "low overall suitability"
            st.markdown(f"""
            <div class="avoid-card">
              <span>🚫</span>
              <div>
                <div class="avoid-crop">{row['Crop'].title()}</div>
                <div style="font-size:0.78rem;color:#c07070;margin-top:2px">Reason: {reason}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<p class="section-label" style="margin-top:20px">Risk Assessment</p>', unsafe_allow_html=True)

    if crop_input != "None":
        row_df = df[(df["District"] == district) & (df["Crop"] == crop_input)]

        if row_df.empty:
            st.markdown('<div class="no-data">No data for this crop in this district.</div>', unsafe_allow_html=True)
        else:
            row = row_df.iloc[0]
            ml_s = predict_score(row_df.iloc[[0]])
            final = 0.6 * row["Suitability_Score"] + 0.4 * ml_s

            if final >= 0.65:
                risk_html = f'<div class="risk-safe">🟢 Safe to Grow &nbsp;·&nbsp; Score {final:.2f}</div>'
                advice = f"{crop_input.title()} is well-suited for {district}. Conditions are favorable this season."
            elif final >= 0.45:
                risk_html = f'<div class="risk-moderate">🟡 Moderate Risk &nbsp;·&nbsp; Score {final:.2f}</div>'
                advice = f"{crop_input.title()} can be grown in {district} with careful irrigation management."
            else:
                risk_html = f'<div class="risk-high">🔴 High Risk &nbsp;·&nbsp; Score {final:.2f}</div>'
                advice = f"{crop_input.title()} is not well-suited for {district} this season — water or soil mismatch."

            st.markdown(f"""
            <div class="risk-panel">
              <p class="risk-title">{crop_input.title()} in {district}</p>
              {risk_html}
              <div class="insight-box" style="margin-top:12px">{advice}</div>
              <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:8px">
                <div class="metric-chip">💧 Water: {match_label(row['Water_Score'])}</div>
                <div class="metric-chip">🌡 Temp: {match_label(row['Temp_Score'])}</div>
                <div class="metric-chip">🌱 Soil: {match_label(row['Soil_Score'])}</div>
                <div class="metric-chip">📦 Yield: {match_label(row['Productivity_Score'])}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="risk-panel">
          <p class="risk-title">Risk Assessment</p>
          <div class="no-data">Select a crop above to check its risk level.</div>
        </div>
        """, unsafe_allow_html=True)

    # District summary card
    if not df_d.empty:
        st.markdown('<p class="section-label" style="margin-top:20px">District Summary</p>', unsafe_allow_html=True)
        avg_rain = df_d["Avg_Rainfall_mm"].iloc[0]
        num_crops = len(df_d)
        best_crop = df_sorted.iloc[0]["Crop"].title() if not df_sorted.empty else "N/A"
        st.markdown(f"""
        <div class="risk-panel">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;text-align:center">
            <div style="background:#f5f2eb;border-radius:10px;padding:14px">
              <div style="font-size:1.4rem;font-weight:600;color:#1a3a2a">{avg_rain:.0f}<span style="font-size:0.75rem;color:#7a8f82"> mm</span></div>
              <div style="font-size:0.75rem;color:#7a8f82;margin-top:2px">Avg Rainfall</div>
            </div>
            <div style="background:#f5f2eb;border-radius:10px;padding:14px">
              <div style="font-size:1.4rem;font-weight:600;color:#1a3a2a">{num_crops}</div>
              <div style="font-size:0.75rem;color:#7a8f82;margin-top:2px">Crops Tracked</div>
            </div>
          </div>
          <div style="margin-top:10px;background:#e8f5e0;border-radius:10px;padding:12px;text-align:center;font-size:0.85rem;color:#1a5c35">
            <span style="font-weight:600">Top pick this season:</span> {best_crop}
          </div>
        </div>
        """, unsafe_allow_html=True)
