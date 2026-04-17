import streamlit as st
import pdfplumber
import docx
import spacy
import io
import re
import plotly.express as px
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ─────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HireMind AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Pipeline Stages (global constant)
# ─────────────────────────────────────────────
PIPELINE_STAGES = ["CV Screening", "Technical Interview", "HR Round", "Offer Letter"]

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(to right, #3944F7, #2530e0) !important;
    }

    /* ── Force ALL text to black ── */
    .stApp, .stApp p, .stApp span, .stApp div,
    .stApp label, .stApp li, .stApp td, .stApp th,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    .stMetric, .stMetric label, .stMetric div {
        color: #000000 !important;
    }

    /* ── Headings ── */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4 {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.10);
        border-right: 1px solid rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
    }

    /* ── Score widget ── */
    .score-ring-wrapper {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        padding: 30px 0;
    }
    .score-number {
        font-size: 4rem; font-weight: 700;
        background: linear-gradient(90deg, #bdf5ff, #28c1fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    .score-label {
        font-size: 1rem; color: #bdf5ff;
        margin-top: 8px; letter-spacing: 0.05em; text-transform: uppercase;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #bdf5ff, #28c1fc) !important;
        border-radius: 8px !important;
    }

    /* ── Badges ── */
    .badge {
        display: inline-block; padding: 4px 12px;
        border-radius: 999px; font-size: 0.78rem; font-weight: 600; margin: 3px;
    }
    .badge-purple { background: rgba(124,58,237,0.6);  color: #ffffff; border: 1px solid rgba(124,58,237,0.8); }
    .badge-blue   { background: rgba(59,130,246,0.6);  color: #ffffff; border: 1px solid rgba(59,130,246,0.8); }
    .badge-green  { background: rgba(16,185,129,0.6);  color: #ffffff; border: 1px solid rgba(16,185,129,0.8); }
    .badge-red    { background: #ef4444;               color: #ffffff; border: 1px solid #dc2626; }
    .badge-orange { background: rgba(245,158,11,0.6);  color: #ffffff; border: 1px solid rgba(245,158,11,0.8); }

    /* ── Section header ── */
    .section-header {
        font-size: 0.75rem; font-weight: 600; color: #bdf5ff !important;
        letter-spacing: 0.10em; text-transform: uppercase; margin-bottom: 12px;
        -webkit-text-fill-color: #bdf5ff !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer { display: none; }

    /* ── Main buttons → Maldives cyan ── */
    .stButton > button,
    [data-testid="baseButton-primary"] {
        background: linear-gradient(to right, #bdf5ff, #28c1fc) !important;
        color: #1a1a2e !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 28px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(40,193,252,0.45);
    }
    .stButton > button:hover,
    [data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 22px rgba(40,193,252,0.65);
        background: linear-gradient(to right, #28c1fc, #bdf5ff) !important;
    }

    /* ── File Uploader 'Browse files' button (secondary) → White ── */
    [data-testid="baseButton-secondary"] {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    [data-testid="baseButton-secondary"]:hover {
        background: #f0f0f0 !important;
    }

    /* ── File uploader zone ── */
    .stFileUploader {
        background: rgba(255,255,255,0.10);
        border: 2px dashed rgba(189,245,255,0.6);
        border-radius: 12px; padding: 16px;
    }
    /* Black text inside uploader */
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] small {
        color: #000000 !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.08) !important;
        border-radius: 8px !important; color: #ffffff !important;
    }

    .stAlert { border-radius: 10px !important; }

    /* ── AI Interpretation ── */
    .interpretation-row { margin-bottom: 12px; line-height: 1.6; color: #ffffff; }
    .interpretation-title { font-weight: 700; color: #bdf5ff; margin-right: 6px; }
    .highlight-green {
        background: #00ff88; color: #000000;
        padding: 2px 6px; border-radius: 4px; font-weight: 800;
        border: 1px solid #00c853; box-shadow: 0 0 5px rgba(0,255,136,0.6);
    }
    .highlight-dim {
        background: rgba(255,255,255,0.08); color: #cbd5e1;
        padding: 2px 6px; border-radius: 4px; font-style: italic;
    }

    /* ── Leaderboard rows ── */
    .lb-row {
        display: flex; align-items: center; justify-content: space-between;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(189,245,255,0.25);
        border-radius: 12px; padding: 14px 20px; margin-bottom: 10px;
        transition: background 0.2s;
    }
    .lb-row:hover { background: rgba(189,245,255,0.18); border-color: rgba(189,245,255,0.5); }
    .lb-rank {
        font-size: 1.4rem; font-weight: 800;
        background: linear-gradient(90deg, #bdf5ff, #28c1fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        min-width: 42px;
    }
    .lb-name { flex: 1; font-weight: 600; color: #ffffff !important; padding: 0 16px; font-size: 0.97rem; }
    .lb-score { font-size: 1.1rem; font-weight: 700; color: #bdf5ff; min-width: 60px; text-align: right; }

    /* ── Selectbox (pipeline stage) ── */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.10) !important;
        border: 1px solid rgba(189,245,255,0.3) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    .stSelectbox label { color: #bdf5ff !important; font-size: 0.75rem !important; }

    /* ── Text input / textarea ── */
    .stTextInput input, .stTextArea textarea {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(189,245,255,0.3) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextInput label, .stTextArea label { color: #bdf5ff !important; }

    /* ── Tab override ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.10);
        border-radius: 12px; padding: 4px; gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #ffffff; border-radius: 8px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(to right, #bdf5ff, #28c1fc) !important;
        color: #1a1a2e !important;
    }

    /* ── Pipeline funnel cards ── */
    .funnel-card {
        padding: 12px 16px; border-radius: 10px;
        margin-bottom: 8px; display: flex;
        align-items: center; justify-content: space-between;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  Load Heavy Models (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP models…")
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource(show_spinner="Loading sentence-transformer…")
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ─────────────────────────────────────────────
#  Helper: Text Extraction
# ─────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    return file_bytes.decode("utf-8", errors="ignore")

# ─────────────────────────────────────────────
#  Helper: Bias Masking (PII Removal)
# ─────────────────────────────────────────────
PRONOUN_MAP = {
    r"\bhe\b": "they", r"\bshe\b": "they",
    r"\bhim\b": "them", r"\bher\b": "them",
    r"\bhis\b": "their", r"\bhers\b": "theirs",
    r"\bhimself\b": "themselves", r"\bherself\b": "themselves"
}

def mask_pii(text: str, candidate_id: str) -> str:
    nlp = load_spacy()
    for pattern, replacement in PRONOUN_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL REDACTED]", text)
    text = re.sub(r"(\+?\d[\d\s\-.()/]{7,}\d)", "[PHONE REDACTED]", text)
    text = re.sub(r"https?://\S+", "[URL REDACTED]", text)
    text = re.sub(r"www\.\S+", "[URL REDACTED]", text)
    text = re.sub(r"linkedin\.com/in/\S+", "[URL REDACTED]", text, flags=re.IGNORECASE)
    text = re.sub(r"github\.com/\S+", "[URL REDACTED]", text, flags=re.IGNORECASE)
    doc = nlp(text)
    replacements = {}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            replacements[ent.text] = f"Candidate {candidate_id}"
        elif ent.label_ in ("GPE", "LOC"):
            replacements[ent.text] = "[LOCATION REDACTED]"
    for original, tag in replacements.items():
        text = text.replace(original, tag)
    return text

# ─────────────────────────────────────────────
#  Helper: Semantic Scoring
# ─────────────────────────────────────────────
def compute_score(resume_text: str, jd_text: str) -> float:
    sbert = load_sbert()
    embeddings = sbert.encode([resume_text, jd_text], convert_to_numpy=True)
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(max(0.0, min(1.0, float(cos_sim))) * 100, 1)

# ─────────────────────────────────────────────
#  Helper: Groq Detailed Interpretation
# ─────────────────────────────────────────────
def get_groq_interpretation(jd_text: str, masked_resume: str) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    prompt = f"""You are an expert technical recruiter. Explain the match score concisely in EXACTLY 4 distinct points.
You must output ONLY this exact HTML list structure, without any intro, outro, or markdown formatting (like **).

<ul style='list-style-type: none; padding-left: 0;'>
    <li class='interpretation-row'><span class='interpretation-title'>🎓 Education:</span> [Analyze education in 1 short sentence. Wrap strong matches in <span class='highlight-green'>...</span> and gaps in <span class='highlight-dim'>...</span>]</li>
    <li class='interpretation-row'><span class='interpretation-title'>💼 Experience:</span> [Analyze experience in 1 short sentence. Wrap strong matches in <span class='highlight-green'>...</span> and gaps in <span class='highlight-dim'>...</span>]</li>
    <li class='interpretation-row'><span class='interpretation-title'>🛠️ Skills:</span> [Analyze skills in 1-2 short sentences. Wrap strong matches in <span class='highlight-green'>...</span> and gaps in <span class='highlight-dim'>...</span>]</li>
    <li class='interpretation-row'><span class='interpretation-title'>📌 Others:</span> [Analyze certifications, projects, or tone. Wrap positives in <span class='highlight-green'>...</span> and negatives in <span class='highlight-dim'>...</span>]</li>
</ul>

--- JOB DESCRIPTION ---
{jd_text}

--- ANONYMISED RESUME ---
{masked_resume}
"""
    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.1,
    )
    return chat_completion.choices[0].message.content.strip()

# ─────────────────────────────────────────────
#  Score Badge Helper
# ─────────────────────────────────────────────
def score_badge(score: float) -> str:
    if score >= 75:
        return '<span class="badge badge-green">STRONG MATCH</span>'
    elif score >= 50:
        return '<span class="badge badge-blue">MODERATE MATCH</span>'
    elif score >= 30:
        return '<span class="badge badge-purple">WEAK MATCH</span>'
    return '<span class="badge badge-red">POOR MATCH</span>'

# ─────────────────────────────────────────────
#  Session State
# ─────────────────────────────────────────────
if "view" not in st.session_state:
    st.session_state.view = "input"
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []
if "selected" not in st.session_state:
    st.session_state.selected = None
if "jd_cache" not in st.session_state:
    st.session_state.jd_cache = ""
# Multi-role: dict of {role_name: jd_text}
if "roles" not in st.session_state:
    st.session_state.roles = {}
# Which role is currently selected for uploading
if "active_role" not in st.session_state:
    st.session_state.active_role = ""
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 8px 0 20px;">
            <div style="font-size:2.4rem;">🤖</div>
            <div style="font-size:1.3rem; font-weight:700;
                        background: linear-gradient(90deg,#bdf5ff,#28c1fc);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                HireMind AI
            </div>
            <div style="font-size:0.78rem; color:#94a3b8; margin-top:4px;">
                Bias-Free · Semantic · AI-Powered
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.view != "input":
        if st.button("➕  Add Roles / Resumes", use_container_width=True):
            st.session_state.view = "input"
            st.session_state.selected = None
            st.session_state.uploader_key += 1
            st.rerun()

    if st.button("🗑️ Start Fresh (Reset All)", use_container_width=True):
        st.session_state.leaderboard = []
        st.session_state.roles = {}
        st.session_state.active_role = ""
        st.session_state.jd_cache = ""
        st.session_state.selected = None
        st.session_state.view = "input"
        st.session_state.uploader_key += 1
        st.rerun()

    if st.session_state.view == "detail" and st.session_state.leaderboard:
        if st.button("🏆  Back to Leaderboard", use_container_width=True):
            st.session_state.view = "leaderboard"
            st.session_state.selected = None
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<p style="color:#64748b;font-size:0.72rem;text-align:center;">Built with ❤️ using Streamlit, spaCy, SBERT & Groq</p>',
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════
tab1, tab2 = st.tabs(["📄  Resume Screening", "📊  Analytics Dashboard"])

# ═════════════════════════════════════════════
#  TAB 1 — MANUAL SCREENER
# ═════════════════════════════════════════════
with tab1:

    # ── VIEW: INPUT ──────────────────────────
    if st.session_state.view == "input":
        st.markdown(
            """
            <h1 style="font-size:2rem; font-weight:700; margin-bottom:4px;
                       background:linear-gradient(90deg,#bdf5ff,#28c1fc);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                HireMind AI
            </h1>
            <p style="color:#94a3b8; margin-bottom:20px;">
                Define a role, paste a job description, upload resumes, and get an unbiased AI-powered ranking.
            </p>
            """,
            unsafe_allow_html=True,
        )

        # ── Role Definition Panel ─────────────
        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.75rem; font-weight:600; color:#bdf5ff; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;">🏷️ Role Management</p>',
            unsafe_allow_html=True,
        )

        role_def_col, role_sel_col = st.columns([2, 1], gap="large")

        with role_def_col:
            with st.expander("➕  Define a New Role", expanded=not bool(st.session_state.roles)):
                new_role_name = st.text_input(
                    "Role Name (e.g. Software Engineer)",
                    placeholder="Enter role title…",
                    key="new_role_name_input",
                )
                new_role_jd = st.text_area(
                    "Job Description for this role",
                    height=150,
                    placeholder="Paste the full JD here…",
                    key="new_role_jd_input",
                )
                if st.button("💾  Save Role", key="save_role_btn"):
                    if new_role_name.strip() and new_role_jd.strip():
                        st.session_state.roles[new_role_name.strip()] = new_role_jd.strip()
                        st.session_state.active_role = new_role_name.strip()
                        st.success(f"✅ Role **{new_role_name.strip()}** saved!")
                        st.rerun()
                    else:
                        st.warning("Please enter both a Role Name and Job Description.")

        with role_sel_col:
            if st.session_state.roles:
                st.markdown(
                    '<p style="font-size:0.75rem; color:#bdf5ff; font-weight:600; text-transform:uppercase; letter-spacing:0.08em;">Saved Roles</p>',
                    unsafe_allow_html=True,
                )
                for rname in list(st.session_state.roles.keys()):
                    is_active = rname == st.session_state.active_role
                    badge_color = "rgba(40,193,252,0.3)" if is_active else "rgba(255,255,255,0.08)"
                    border_color = "#28c1fc" if is_active else "rgba(255,255,255,0.15)"
                    st.markdown(
                        f"""<div style="background:{badge_color}; border:1px solid {border_color};
                                        border-radius:8px; padding:8px 12px; margin-bottom:6px;
                                        cursor:pointer; font-size:0.88rem; color:#ffffff;">
                                {"✅ " if is_active else ""}{rname}
                            </div>""",
                        unsafe_allow_html=True,
                    )

        # Active role selector
        if st.session_state.roles:
            active_role = st.selectbox(
                "📌 Select Role for this upload batch",
                options=list(st.session_state.roles.keys()),
                index=list(st.session_state.roles.keys()).index(st.session_state.active_role)
                      if st.session_state.active_role in st.session_state.roles else 0,
                key="active_role_select",
            )
            st.session_state.active_role = active_role
            jd_text = st.session_state.roles[active_role]
            st.markdown(
                f'<p style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">Using JD for: <strong style="color:#28c1fc;">{active_role}</strong></p>',
                unsafe_allow_html=True,
            )
        else:
            st.info("💡 Define at least one Role above before uploading resumes.")
            jd_text = ""

        st.markdown("---")

        # ── Resume Upload ─────────────────────
        st.markdown(
            '<p style="font-size:0.75rem; font-weight:600; color:#bdf5ff; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px;">📄 Resume Upload</p>',
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Drop resumes here (PDF or DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            label_visibility="visible",
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if uploaded_files:
            for f in uploaded_files:
                st.markdown(f'<span class="badge badge-green">✓ {f.name}</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([2, 1, 2])
        with btn_col:
            analyse_btn = st.button("🚀  Analyse Resume(s)", use_container_width=True, key="manual_analyse")

        st.markdown("---")

        if analyse_btn:
            if not st.session_state.roles:
                st.error("❌ Please define at least one Role before analysing.")
            elif not jd_text.strip():
                st.error("❌ No Job Description found for the selected role.")
            elif not uploaded_files:
                st.error("❌ Please upload at least one resume (PDF or DOCX).")
            else:
                st.session_state.jd_cache = jd_text
                results = []
                
                catchy_phrases = [
                    "Summoning the AI recruiters...",
                    "Scanning the matrix for top talent...",
                    "Crunching the candidate data...",
                    "Extracting semantic signals...",
                    "Filtering the elite from the crowd..."
                ]
                
                progress_bar = st.progress(0, text=random.choice(catchy_phrases))

                for idx, uf in enumerate(uploaded_files):
                    file_bytes = uf.read()
                    original_filename = uf.name
                    candidate_id = f"#{idx + 1}"
                    display_name = f"Candidate {candidate_id}"

                    with st.spinner(f"Processing {display_name}…"):
                        raw_text = extract_text_from_bytes(file_bytes, original_filename)

                    if not raw_text.strip():
                        st.warning(f"⚠️ Could not extract text from {original_filename} — skipping.")
                        continue

                    masked_text = mask_pii(raw_text, candidate_id)
                    score       = compute_score(masked_text, jd_text)

                    results.append({
                        "filename":    display_name,
                        "score":       score,
                        "masked_text": masked_text,
                        "ai_summary":  None,
                        "rank":        None,
                        "role":        st.session_state.active_role,
                        "stage":       PIPELINE_STAGES[0],  # default: CV Screening
                    })

                    progress_bar.progress(
                        int((idx + 1) / len(uploaded_files) * 100),
                        text=f"Processed {idx + 1}/{len(uploaded_files)}: {display_name}",
                    )

                progress_bar.empty()

                if not results:
                    st.error("No resumes could be processed.")
                    st.stop()

                st.session_state.leaderboard.extend(results)

                # Temporarily filter/rank so detail view or immediate leaderboard handles just added resumes correctly
                active_lb = [r for r in st.session_state.leaderboard if r.get("role") == st.session_state.active_role]
                active_lb.sort(key=lambda r: r["score"], reverse=True)
                for i, r in enumerate(active_lb):
                    r["rank"] = i + 1

                if len(results) == 1:
                    with st.spinner("🤖 Generating AI interpretation…"):
                        try:
                            results[0]["ai_summary"] = get_groq_interpretation(jd_text, results[0]["masked_text"])
                        except Exception as exc:
                            st.warning(f"Groq API error: {exc}")
                    st.session_state.selected = results[0]
                    st.session_state.view = "detail"
                else:
                    st.session_state.view = "leaderboard"

                st.rerun()

    # ── VIEW: LEADERBOARD ────────────────────
    elif st.session_state.view == "leaderboard":
        
        role_filter = st.selectbox(
            "Filter Leaderboard by Role",
            options=list(st.session_state.roles.keys()),
            index=list(st.session_state.roles.keys()).index(st.session_state.active_role) if st.session_state.active_role in st.session_state.roles else 0,
            key="leaderboard_role_filter"
        )
        
        lb = [r for r in st.session_state.leaderboard if r.get("role") == role_filter]
        
        # Sort and assign rank dynamically for this filtered list
        lb.sort(key=lambda r: r["score"], reverse=True)
        for i, r in enumerate(lb):
            r["rank"] = i + 1
            
        jd_text = st.session_state.roles.get(role_filter, "")

        st.markdown(
            """
            <h1 style="font-size:2rem; font-weight:700; margin-bottom:4px;
                       background:linear-gradient(90deg,#bdf5ff,#28c1fc);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                🏆 Candidate Leaderboard
            </h1>
            <p style="color:#94a3b8; margin-bottom:28px;">
                Ranked by semantic fit. Use the stage selector on each row to advance candidates through the pipeline.
            </p>
            """,
            unsafe_allow_html=True,
        )

        total     = len(lb)
        top_score = lb[0]["score"] if lb else 0
        strong    = sum(1 for r in lb if r["score"] >= 75)
        weak      = sum(1 for r in lb if r["score"] < 50)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Candidates", total)
        s2.metric("Top Score", f"{top_score}%")
        s3.metric("Strong Matches", strong)
        s4.metric("Weak / Poor", weak)

        st.markdown("---")

        for item in lb:
            rank  = item["rank"]
            name  = item["filename"]
            score = item["score"]
            role  = item.get("role", "—")

            # ── Row: score info + stage selector + view button
            row_col, stage_col, btn_col = st.columns([5, 3, 1], gap="small")

            with row_col:
                st.markdown(
                    f"""
                    <div class="lb-row">
                        <div class="lb-rank">#{rank}</div>
                        <div class="lb-name">{name}
                            <span style="font-size:0.72rem; font-weight:400; color:#94a3b8; margin-left:8px;">
                                {role}
                            </span>
                        </div>
                        {score_badge(score)}
                        <div class="lb-score">{score}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with stage_col:
                current_stage = item.get("stage", PIPELINE_STAGES[0])
                new_stage = st.selectbox(
                    "Pipeline Stage",
                    options=PIPELINE_STAGES,
                    index=PIPELINE_STAGES.index(current_stage),
                    key=f"stage_{rank}",
                    label_visibility="collapsed",
                )
                if new_stage != current_stage:
                    item["stage"] = new_stage
                    st.rerun()

            with btn_col:
                if st.button("View →", key=f"view_{rank}"):
                    if item["ai_summary"] is None:
                        with st.spinner(f"🤖 Generating AI interpretation for {name}…"):
                            try:
                                item["ai_summary"] = get_groq_interpretation(jd_text, item["masked_text"])
                            except Exception as exc:
                                item["ai_summary"] = f"<p>Groq API error: {exc}</p>"
                    st.session_state.selected = item
                    st.session_state.view = "detail"
                    st.rerun()

    # ── VIEW: DETAIL ─────────────────────────
    elif st.session_state.view == "detail":
        item = st.session_state.selected
        if item is None:
            st.session_state.view = "input"
            st.rerun()

        score      = item["score"]
        ai_summary = item["ai_summary"]
        filename   = item["filename"]
        rank       = item["rank"]
        role       = item.get("role", "—")
        stage      = item.get("stage", PIPELINE_STAGES[0])
        role_candidates = [r for r in st.session_state.leaderboard if r.get("role") == role]
        is_bulk    = len(role_candidates) > 1

        rank_label = f"Rank #{rank} of {len(role_candidates)}" if is_bulk else ""
        st.markdown(
            f"""
            <h1 style="font-size:1.8rem; font-weight:700; margin-bottom:2px;
                       background:linear-gradient(90deg,#bdf5ff,#28c1fc);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                📋 Detailed Analysis
            </h1>
            <p style="color:#94a3b8; margin-bottom:6px;">
                <strong style="color:#28c1fc;">{filename}</strong>
                &nbsp;·&nbsp;<span style="color:#bdf5ff;">{role}</span>
                {"&nbsp;&nbsp;·&nbsp;&nbsp;" + rank_label if rank_label else ""}
            </p>
            """,
            unsafe_allow_html=True,
        )

        # Pipeline stage selector on detail page
        detail_stage = st.selectbox(
            "🔄 Pipeline Stage",
            options=PIPELINE_STAGES,
            index=PIPELINE_STAGES.index(stage),
            key="detail_stage_select",
        )
        if detail_stage != stage:
            item["stage"] = detail_stage
            st.rerun()

        st.markdown("---")

        res_col1, res_col2 = st.columns([1, 2], gap="large")

        with res_col1:
            st.markdown(
                f"""
                <div class="score-ring-wrapper">
                    <div class="score-number">{score}%</div>
                    <div class="score-label">Match Score</div>
                    {score_badge(score)}
                    {"<br><span style='color:#bdf5ff;font-size:0.85rem; margin-top:15px;'>Rank #" + str(rank) + " of " + str(len(role_candidates)) + "</span>" if is_bulk else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(int(score))

        with res_col2:
            st.markdown(
                '<p class="section-header">🤖 AI Scoring Reasons</p>',
                unsafe_allow_html=True,
            )
            if ai_summary:
                st.markdown(
                    f'<div style="font-size:0.95rem; color:#ffffff;">{ai_summary}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("AI interpretation unavailable.")


# ═════════════════════════════════════════════
#  TAB 2 — ANALYTICS DASHBOARD
# ═════════════════════════════════════════════
with tab2:
    st.markdown(
        """
        <h1 style="font-size:2rem; font-weight:700; margin-bottom:4px;
                   background:linear-gradient(90deg,#bdf5ff,#28c1fc);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            📊 Analytics Dashboard
        </h1>
        <p style="color:#94a3b8; margin-bottom:24px;">
            Real-time pipeline insights across all roles and candidates.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.leaderboard:
        st.info("💡 Run **HireMind AI** (Tab 1) first to populate the dashboard.")
    else:
        dash_role = st.selectbox(
            "View Analytics for Role",
            options=["All Roles"] + list(st.session_state.roles.keys()),
            key="dash_role_filter"
        )
        
        if dash_role == "All Roles":
            lb = st.session_state.leaderboard
        else:
            lb = [r for r in st.session_state.leaderboard if r.get("role") == dash_role]
            
        if not lb:
            st.warning(f"No candidates found for {dash_role}.")
        else:
            scores   = [r["score"] for r in lb]
        total    = len(scores)
        strong   = sum(1 for s in scores if s >= 75)
        moderate = sum(1 for s in scores if 50 <= s < 75)
        weak     = sum(1 for s in scores if 30 <= s < 50)
        poor     = sum(1 for s in scores if s < 30)
        avg      = round(sum(scores) / total, 1)

        # ── KPI strip ─────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Candidates", total)
        k2.metric("Avg Score", f"{avg}%")
        k3.metric("🟢 Strong", strong)
        k4.metric("🔵 Moderate", moderate)
        k5.metric("🔴 Weak / Poor", weak + poor)

        st.markdown("---")

        # ── Row 1: Pie Chart (Applicants/Role) + Score Tier ────
        pie_col, tier_col = st.columns(2, gap="large")

        with pie_col:
            st.markdown("#### 🏷️ Applicants per Role")
            role_counts: dict[str, int] = {}
            for r in lb:
                role_name = r.get("role", "Unknown")
                role_counts[role_name] = role_counts.get(role_name, 0) + 1

            if role_counts:
                pie_fig = px.pie(
                    names=list(role_counts.keys()),
                    values=list(role_counts.values()),
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    hole=0.4,
                )
                pie_fig.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#000000", family="Inter"),
                    margin=dict(t=20, b=20, l=10, r=10),
                    legend=dict(font=dict(color="#000000")),
                    showlegend=True,
                )
                pie_fig.update_traces(textfont_color="#000000")
                st.plotly_chart(pie_fig, use_container_width=True)
            else:
                st.info("No role data available.")

        with tier_col:
            st.markdown("#### 🎯 Score Tier Distribution")
            tier_fig = px.bar(
                x=["Strong (≥75%)", "Moderate (50–74%)", "Weak (30–49%)", "Poor (<30%)"],
                y=[strong, moderate, weak, poor],
                color=["Strong (≥75%)", "Moderate (50–74%)", "Weak (30–49%)", "Poor (<30%)"],
                color_discrete_map={
                    "Strong (≥75%)":    "#00ffaa",
                    "Moderate (50–74%)": "#00aaff",
                    "Weak (30–49%)":    "#ffaa00",
                    "Poor (<30%)":      "#ff0055",
                },
                text=[strong, moderate, weak, poor]
            )
            tier_fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#000000", family="Inter", size=14),
                margin=dict(t=20, b=20, l=10, r=10),
                showlegend=False,
                xaxis=dict(showgrid=False, title="", tickcolor="#000000", tickfont=dict(color="#000000")),
                yaxis=dict(showgrid=False, title="", tickcolor="#000000", tickfont=dict(color="#000000")),
            )
            tier_fig.update_traces(textposition="outside", textfont=dict(color="#000000", size=14))
            st.plotly_chart(tier_fig, use_container_width=True)

        st.markdown("---")

        # ── Row 2: Funnel Chart + Candidate Table ──────────────
        funnel_col, table_col = st.columns([1, 2], gap="large")

        with funnel_col:
            st.markdown("#### 🔽 Hiring Pipeline Funnel")

            stage_counts = {stage: 0 for stage in PIPELINE_STAGES}
            for r in lb:
                s = r.get("stage", PIPELINE_STAGES[0])
                if s in stage_counts:
                    stage_counts[s] += 1

            # Build cumulative funnel values (each stage includes those below it)
            funnel_values = []
            cumulative = total
            for stage in PIPELINE_STAGES:
                funnel_values.append(cumulative)
                cumulative -= stage_counts[stage]

            funnel_fig = px.funnel(
                y=PIPELINE_STAGES,
                x=funnel_values,
                color=PIPELINE_STAGES,
                color_discrete_sequence=["#00aaff", "#00ffaa", "#ffaa00", "#ff0055"],
            )
            funnel_fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#000000", family="Inter", size=14),
                margin=dict(t=20, b=20, l=20, r=10),
                showlegend=False,
                yaxis=dict(showgrid=False, title="", tickcolor="#000000", tickfont=dict(color="#000000")),
                xaxis=dict(showgrid=False, showticklabels=False, title="", tickcolor="#000000", tickfont=dict(color="#000000")),
            )
            funnel_fig.update_traces(
                textinfo="value", 
                textposition="inside", 
                textfont=dict(color="#000000", size=14), 
                insidetextfont=dict(color="#000000", size=14)
            )
            st.plotly_chart(funnel_fig, use_container_width=True)

            # Bright stage cards below funnel
            stage_colors = [
                ("#0c4a6e", "#00aaff", "#000000"),
                ("#064e3b", "#00ffaa", "#000000"),
                ("#451a03", "#ffaa00", "#000000"),
                ("#450a0a", "#ff0055", "#000000"),
            ]
            for (stage, count), (border, bg, text) in zip(stage_counts.items(), stage_colors):
                st.markdown(
                    f"""<div style="background:{bg}aa; border-left:4px solid {border};
                                    padding:10px 14px; border-radius:8px; margin-bottom:6px;
                                    display:flex; justify-content:space-between; align-items:center;">
                            <strong style="color:{text};">{stage}</strong>
                            <span style="font-weight:700; color:#ffffff; font-size:1.1rem;">{count}</span>
                        </div>""",
                    unsafe_allow_html=True,
                )

        with table_col:
            st.markdown("#### 📋 Candidate Score Table")
            rows_html = ""
            for r in lb:
                stage_badge_colors = {
                    "CV Screening":       ("#28c1fc", "#0c4a6e"),
                    "Technical Interview": ("#10b981", "#064e3b"),
                    "HR Round":           ("#f59e0b", "#451a03"),
                    "Offer Letter":       ("#ef4444", "#450a0a"),
                }
                sc = r.get("stage", PIPELINE_STAGES[0])
                sc_color, sc_bg = stage_badge_colors.get(sc, ("#ffffff", "#1e293b"))
                rows_html += f"""
                <tr style="border-bottom:1px solid rgba(255,255,255,0.08);">
                    <td style="padding:8px 12px; color:#bdf5ff; font-weight:700;">#{r['rank']}</td>
                    <td style="padding:8px 12px; color:#ffffff;">{r['filename']}</td>
                    <td style="padding:8px 12px; color:#94a3b8; font-size:0.82rem;">{r.get('role','—')}</td>
                    <td style="padding:8px 12px;">{score_badge(r['score'])}</td>
                    <td style="padding:8px 12px;">
                        <span style="background:{sc_bg}cc; color:{sc_color}; padding:3px 10px;
                                     border-radius:999px; font-size:0.75rem; font-weight:600;
                                     border:1px solid {sc_color}60;">{sc}</span>
                    </td>
                    <td style="padding:8px 12px; color:#bdf5ff; font-weight:700; text-align:right;">{r['score']}%</td>
                </tr>"""

            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.06); border-radius:12px;
                            border:1px solid rgba(189,245,255,0.15); overflow:hidden;">
                    <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
                        <thead>
                            <tr style="background:rgba(40,193,252,0.15);">
                                <th style="padding:10px 12px; text-align:left; color:#bdf5ff;">Rank</th>
                                <th style="padding:10px 12px; text-align:left; color:#bdf5ff;">Candidate</th>
                                <th style="padding:10px 12px; text-align:left; color:#bdf5ff;">Role</th>
                                <th style="padding:10px 12px; text-align:left; color:#bdf5ff;">Tier</th>
                                <th style="padding:10px 12px; text-align:left; color:#bdf5ff;">Stage</th>
                                <th style="padding:10px 12px; text-align:right; color:#bdf5ff;">Score</th>
                            </tr>
                        </thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )