import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go

# ---------- Config ----------
st.set_page_config(page_title="IntroSpeech AI", layout="wide", initial_sidebar_state="collapsed")

# Professional dark theme colors
PRIMARY = "#8b5cf6"
ACCENT = "#a78bfa"
SUCCESS = "#10b981"
WARNING = "#f59e0b"
ERROR = "#ef4444"
DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
CARD_HOVER = "#334155"
TEXT = "#f1f5f9"
MUTED = "#94a3b8"
BORDER = "#334155"

# ---------- Enhanced CSS ----------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, #root, .reportview-container, .main, .block-container {{
      background: {DARK_BG} !important;
      color: {TEXT};
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .stApp {{
      background: {DARK_BG};
    }}
    
    /* Hide Streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Hero Header */
    .hero-header {{
      background: linear-gradient(135deg, {PRIMARY} 0%, #6366f1 100%);
      padding: 60px 40px;
      border-radius: 24px;
      margin-bottom: 40px;
      box-shadow: 0 25px 50px -12px rgba(139, 92, 246, 0.25);
      text-align: center;
      position: relative;
      overflow: hidden;
    }}
    
    .hero-header::before {{
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: radial-gradient(circle at 30% 50%, rgba(167, 139, 250, 0.2), transparent 50%);
    }}
    
    .hero-title {{
      font-size: 52px;
      font-weight: 800;
      color: white;
      margin-bottom: 12px;
      letter-spacing: -1.5px;
      position: relative;
      z-index: 1;
    }}
    
    .hero-subtitle {{
      font-size: 20px;
      color: rgba(255, 255, 255, 0.95);
      font-weight: 500;
      position: relative;
      z-index: 1;
    }}
    
    /* Section Headers */
    .section-header {{
      font-size: 28px;
      font-weight: 700;
      color: {TEXT};
      margin: 40px 0 24px 0;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    
    .section-icon {{
      width: 36px;
      height: 36px;
      background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
    }}
    
    /* Card styles */
    
    
    .card:hover {{
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.3);
      transform: translateY(-2px);
      border-color: {PRIMARY};
    }}
    
    .card-title {{
      font-size: 20px;
      font-weight: 700;
      color: {TEXT};
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    
    .card-icon {{
      width: 44px;
      height: 44px;
      background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
    }}
    
    /* Metrics */
    .metric-card {{
      background: {CARD_BG};
      padding: 32px 24px;
      border-radius: 20px;
      text-align: center;
      border: 1px solid {BORDER};
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
      transition: all 0.3s ease;
      position: relative;
      overflow: hidden;
    }}
    
    .metric-card::before {{
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 4px;
      background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
    }}
    
    .metric-card:hover {{
      transform: translateY(-4px);
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
    }}
    
    .metric-value {{
      font-size: 42px;
      font-weight: 800;
      background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin: 12px 0;
    }}
    
    .metric-label {{
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: {MUTED};
      margin-bottom: 8px;
    }}
    
    .metric-subtitle {{
      font-size: 14px;
      color: {MUTED};
      margin-top: 8px;
    }}
    
    /* Button */
    .stButton>button {{
      background: linear-gradient(135deg, {PRIMARY}, #6366f1);
      color: white;
      border-radius: 14px;
      padding: 16px 40px;
      font-weight: 700;
      font-size: 17px;
      border: none;
      box-shadow: 0 10px 30px rgba(139, 92, 246, 0.4);
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      width: 100%;
      letter-spacing: 0.5px;
    }}
    
    .stButton>button:hover {{
      transform: translateY(-3px);
      box-shadow: 0 20px 40px rgba(139, 92, 246, 0.5);
    }}
    
    /* Text area */
    .stTextArea textarea {{
      background: {CARD_BG} !important;
      border: 2px solid {BORDER} !important;
      border-radius: 14px !important;
      padding: 18px !important;
      font-size: 15px !important;
      color: {TEXT} !important;
      transition: all 0.3s ease !important;
      line-height: 1.6 !important;
    }}
    
    .stTextArea textarea:focus {{
      border-color: {PRIMARY} !important;
      box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2) !important;
      outline: none !important;
    }}
    
    .stTextArea label {{
      color: {TEXT} !important;
      font-weight: 600 !important;
      margin-bottom: 8px !important;
    }}
    
    /* File uploader */
    .stFileUploader {{
      background: {CARD_BG};
      border: 2px dashed {BORDER};
      border-radius: 14px;
      padding: 28px;
      transition: all 0.3s ease;
    }}
    
    .stFileUploader:hover {{
      border-color: {PRIMARY};
      background: {CARD_HOVER};
    }}
    
    .stFileUploader label {{
      color: {TEXT} !important;
      font-weight: 600 !important;
    }}
    
    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {{
      background: {CARD_BG} !important;
      border-radius: 12px !important;
      border-left: 4px solid {SUCCESS} !important;
      padding: 16px !important;
      color: {TEXT} !important;
    }}
    
    .stError {{
      border-left-color: {ERROR} !important;
    }}
    
    .stWarning {{
      border-left-color: {WARNING} !important;
    }}
    
    /* Keyword chips */
    .keyword-chip {{
      display: inline-block;
      padding: 8px 16px;
      border-radius: 20px;
      background: rgba(139, 92, 246, 0.15);
      color: {ACCENT};
      margin: 4px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid rgba(139, 92, 246, 0.3);
      transition: all 0.2s ease;
    }}
    
    .keyword-chip:hover {{
      background: rgba(139, 92, 246, 0.25);
      transform: scale(1.05);
    }}
    
    /* Score badge */
    .score-badge {{
      display: inline-block;
      padding: 12px 24px;
      border-radius: 14px;
      font-weight: 800;
      font-size: 24px;
      background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
      color: white;
      box-shadow: 0 8px 16px rgba(139, 92, 246, 0.3);
    }}
    
    /* Criterion card */
    .criterion-card {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 20px;
      transition: all 0.3s ease;
    }}
    
    .criterion-card:hover {{
      border-color: {PRIMARY};
      box-shadow: 0 8px 24px rgba(139, 92, 246, 0.2);
      transform: translateX(4px);
    }}
    
    .criterion-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid {BORDER};
    }}
    
    .criterion-title {{
      font-size: 19px;
      font-weight: 700;
      color: {TEXT};
    }}
    
    .criterion-weight {{
      background: rgba(139, 92, 246, 0.2);
      color: {ACCENT};
      padding: 6px 14px;
      border-radius: 10px;
      font-size: 13px;
      font-weight: 700;
      border: 1px solid rgba(139, 92, 246, 0.3);
    }}
    
    .criterion-description {{
      color: {MUTED};
      line-height: 1.6;
      margin-bottom: 16px;
      font-size: 14px;
    }}
    
    .criterion-stats {{
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid {BORDER};
      font-size: 13px;
      color: {MUTED};
    }}
    
    .criterion-stats strong {{
      color: {TEXT};
    }}
    
    /* Info box */
    .info-box {{
      background: rgba(139, 92, 246, 0.1);
      border-left: 4px solid {PRIMARY};
      padding: 20px;
      border-radius: 12px;
      margin: 20px 0;
      color: {TEXT};
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
      background: {CARD_BG} !important;
      border-radius: 12px !important;
      color: {TEXT} !important;
      font-weight: 600 !important;
      border: 1px solid {BORDER} !important;
    }}
    
    .streamlit-expanderHeader:hover {{
      border-color: {PRIMARY} !important;
    }}
    
    .streamlit-expanderContent {{
      background: {CARD_BG} !important;
      border: 1px solid {BORDER} !important;
      border-top: none !important;
      border-radius: 0 0 12px 12px !important;
    }}
    
    /* DataFrame */
    .dataframe {{
      background: {CARD_BG} !important;
      color: {TEXT} !important;
    }}
    
    /* Footer */
    .footer {{
      text-align: center;
      padding: 40px 20px;
      color: {MUTED};
      font-size: 14px;
      margin-top: 60px;
      border-top: 1px solid {BORDER};
    }}
    
    .footer-brand {{
      font-weight: 700;
      background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    
    /* Download button */
    .stDownloadButton>button {{
      background: {CARD_BG} !important;
      color: {TEXT} !important;
      border: 2px solid {PRIMARY} !important;
      border-radius: 12px !important;
      padding: 12px 24px !important;
      font-weight: 600 !important;
      transition: all 0.3s ease !important;
    }}
    
    .stDownloadButton>button:hover {{
      background: {PRIMARY} !important;
      transform: translateY(-2px) !important;
    }}
    
    /* Plotly charts */
    .js-plotly-plot {{
      background: transparent !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Hero Header ----------
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">IntroSpeech AI</div>
        <div class="hero-subtitle">AI-Powered Student Introduction Evaluation System</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Model load ----------
@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = get_model()

# ---------- helpers ----------
def tokenize(text):
    return re.findall(r"\w+", text.lower())

def normalize_cosine(x, low=0.18, high=0.9):
    x = float(x)
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return (x - low) / (high - low)

def keyword_score(keywords, tokens):
    if not keywords or str(keywords).strip() == "":
        return 0.0, []
    kws = [k.strip().lower() for k in str(keywords).split(",") if k.strip()]
    if len(kws) == 0:
        return 0.0, []
    found = []
    text = " ".join(tokens)
    for k in kws:
        k_clean = re.sub(r"\W+", " ", k)
        if k_clean in text:
            found.append(k)
        else:
            k_tokens = set(re.findall(r"\w+", k))
            if k_tokens & set(tokens):
                found.append(k)
    return len(found) / len(kws), found

def compute_scores(transcript, rubric_df):
    t_tokens = tokenize(transcript)
    t_emb = model.encode(transcript, convert_to_tensor=True)
    word_count = len(t_tokens)

    rows = []
    for _, r in rubric_df.iterrows():
        crit = str(r.get("criterion", "")).strip()
        desc = str(r.get("description", "")).strip()
        keywords = r.get("keywords", "") if "keywords" in r else ""
        weight = float(r.get("weight", 1) or 1)

        k_score, matched = keyword_score(keywords, t_tokens)
        desc_emb = model.encode(desc, convert_to_tensor=True)
        cos = util.cos_sim(desc_emb, t_emb).item()
        s_score = normalize_cosine(cos)

        struct = 0.0
        txt = transcript.lower()
        if any(x in txt for x in ["hello", "hi", "good morning", "good evening"]): struct += 0.25
        if any(x in txt for x in ["my name", "i am", "myself"]): struct += 0.25
        if word_count > 8: struct += 0.25
        if "thank" in txt: struct += 0.25

        combined = 0.35 * k_score + 0.50 * s_score + 0.15 * struct
        score = combined * 100.0

        rows.append({
            "criterion": crit,
            "description": desc,
            "keywords": keywords,
            "weight": weight,
            "score": round(score, 2),
            "matched": matched,
            "similarity": round(cos, 3),
            "structure": round(struct, 3)
        })

    df_res = pd.DataFrame(rows)
    overall = 0.0
    if len(df_res) > 0:
        weights = df_res["weight"].to_numpy(dtype=float)
        scores = df_res["score"].to_numpy(dtype=float)
        overall = (weights * scores).sum() / weights.sum()
    return df_res, round(float(overall), 2)

# ---------- Main Input Section ----------
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><div class="card-icon">📋</div>Upload Rubric</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose your rubric file (.xlsx)", type=["xlsx"], label_visibility="collapsed")
    
    if uploaded:
        st.markdown('<div class="info-box">Rubric file uploaded successfully</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><div class="card-icon">✍️</div>Student Transcript</div>', unsafe_allow_html=True)
    transcript = st.text_area(
        "Paste the student's introduction speech here",
        height=200,
        placeholder="Example: Hello everyone! My name is John Smith. I'm a computer science student passionate about artificial intelligence...",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Load rubric ----------
rubric_df = pd.DataFrame(columns=["criterion", "description", "keywords", "weight"])
if uploaded is not None:
    try:
        rubric_df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded rubric: {str(e)}")

# ---------- Score Button ----------
st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

score_btn_col1, score_btn_col2, score_btn_col3 = st.columns([1, 2, 1])
with score_btn_col2:
    analyze_button = st.button("Analyze Speech", use_container_width=True)

if analyze_button:
    if rubric_df.empty:
        st.error("Please upload a rubric file to continue.")
    elif transcript.strip() == "":
        st.error("Please paste a transcript to analyze.")
    else:
        with st.spinner("Analyzing transcript with AI..."):
            df_res, overall = compute_scores(transcript, rubric_df)
            time.sleep(0.5)

        # ---------- Results Section ----------
        st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><div class="section-icon">📊</div>Performance Metrics</div>', unsafe_allow_html=True)
        
        # Top Metrics
        m1, m2, m3 = st.columns(3, gap="large")
        
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Overall Score</div>
                    <div class="metric-value">{overall}</div>
                    <div class="metric-subtitle">out of 100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with m2:
            avg_sim = df_res["similarity"].mean() if not df_res.empty else 0.0
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Similarity</div>
                    <div class="metric-value">{round(avg_sim, 3)}</div>
                    <div class="metric-subtitle">semantic match</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Criteria Evaluated</div>
                    <div class="metric-value">{len(df_res)}</div>
                    <div class="metric-subtitle">total criteria</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Charts Section
        st.markdown('<div style="height:40px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><div class="section-icon">📈</div>Visual Analysis</div>', unsafe_allow_html=True)
        
        chart_col1, chart_col2 = st.columns([2, 1], gap="large")
        
        with chart_col1:
            if not df_res.empty:
                bar_df = df_res.sort_values("score", ascending=True)
                fig = px.bar(
                    bar_df,
                    x="score",
                    y="criterion",
                    orientation="h",
                    labels={"score": "Score", "criterion": "Criterion"},
                    color="score",
                    color_continuous_scale="Purples",
                    height=450
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=20, b=10),
                    font=dict(family="Inter", size=12, color=TEXT),
                    showlegend=False,
                    xaxis=dict(gridcolor=BORDER),
                    yaxis=dict(gridcolor=BORDER)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with chart_col2:
            if not df_res.empty:
                pie_df = df_res.copy()
                pie_df["contrib"] = pie_df["weight"] * pie_df["score"]
                if pie_df["contrib"].sum() > 0:
                    figp = px.pie(
                        pie_df,
                        names="criterion",
                        values="contrib",
                        color_discrete_sequence=px.colors.sequential.Purples,
                        height=450
                    )
                    figp.update_traces(textposition='inside', textinfo='percent')
                    figp.update_layout(
                        margin=dict(l=10, r=10, t=20, b=10),
                        font=dict(family="Inter", size=11, color=TEXT),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(figp, use_container_width=True)

        # Detailed Results
        st.markdown('<div style="height:40px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><div class="section-icon">📝</div>Detailed Criterion Breakdown</div>', unsafe_allow_html=True)
        
        for _, row in df_res.iterrows():
            col_a, col_b = st.columns([4, 1], gap="medium")
            
            with col_a:
                st.markdown(
                    f"""
                    <div class="criterion-card">
                        <div class="criterion-header">
                            <div class="criterion-title">{row["criterion"]}</div>
                            <div class="criterion-weight">Weight: {row["weight"]}</div>
                        </div>
                        <div class="criterion-description">{row["description"]}</div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Keywords
                kws = row["keywords"] if row["keywords"] else ""
                if kws:
                    chips = ""
                    for k in [x.strip() for x in str(kws).split(",") if x.strip()]:
                        chips += f'<span class="keyword-chip">{k}</span>'
                    st.markdown(f'<div style="margin: 16px 0;">{chips}</div>', unsafe_allow_html=True)
                
                # Matched info
                matched = ", ".join(row["matched"]) if row["matched"] else "None"
                st.markdown(
                    f"""
                        <div class="criterion-stats">
                            <strong>Matched keywords:</strong> {matched}<br>
                            <strong>Semantic similarity:</strong> {row["similarity"]} | 
                            <strong>Structure score:</strong> {row["structure"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col_b:
                st.markdown(
                    f"""
                    <div class="card" style="text-align: center; padding: 28px 20px;">
                        <div style="color: {MUTED}; font-size: 13px; font-weight: 600; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">
                            Score
                        </div>
                        <div class="score-badge">{row["score"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Expandable section
        st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)
        with st.expander("View Full Transcript & Download Results"):
            st.markdown("**Original Transcript:**")
            st.write(transcript)
            st.markdown("---")
            st.markdown("**Detailed Scoring Data:**")
            st.dataframe(df_res, use_container_width=True)
            
            csv = df_res.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results as CSV",
                data=csv,
                file_name="introspeech_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.success("Analysis completed successfully")

# ---------- Footer ----------
st.markdown(
    """
    <div class="footer">
        <div>Powered by <span class="footer-brand">IntroSpeech AI</span></div>
        <div style="margin-top: 8px; font-size: 12px;">
            Built with Streamlit • Sentence Transformers • Plotly
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)