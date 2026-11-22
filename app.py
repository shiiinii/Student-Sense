"""Student Success Predictor — Streamlit interface.

The model and original feature schema are preserved.  The academic outcome
shown here is a proxy for the dataset's High/Low activity rating.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Student Success | Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {--ink:#17263d;--muted:#6f7e92;--border:#e5ebf1;--teal:#117d77;--soft:#f4f7fa;--navy:#0f2339;}
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background: #f6f8fb; color:var(--ink);}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stMainBlockContainer"] {max-width:1240px;padding:1.1rem 2.2rem 4.5rem;}
[data-testid="stAppViewContainer"] h1,[data-testid="stAppViewContainer"] h2,[data-testid="stAppViewContainer"] h3 {font-family:'Manrope',sans-serif;letter-spacing:-.035em;}
[data-testid="stSidebar"] {background:#f6f8fb;}
#MainMenu, footer {visibility:hidden;}
/* App header */
.navbar {display:flex;justify-content:space-between;align-items:center;gap:16px;padding:8px 0 23px;}
.brand {display:flex;align-items:center;gap:11px;font-weight:800;font-size:19px;letter-spacing:-.8px;color:#17283e;}
.brand-symbol {width:39px;height:39px;border-radius:13px;background:#117d77;display:grid;place-items:center;color:white;font-size:22px;box-shadow:0 7px 13px rgba(17,125,119,.19);}
.brand span.text-muted {color:#117d77;}
.nav-right {display:flex;align-items:center;gap:9px;color:#6f7e92;font-size:12px;font-weight:700;}
.nav-status {width:8px;height:8px;background:#30b39a;border-radius:50%;box-shadow:0 0 0 4px #e0f6ef;}
.nav-chip {margin-left:12px;border:1px solid #e3eaf0;background:white;padding:7px 12px;border-radius:100px;color:#4c6174;}
/* Hero */
.hero {position:relative;isolation:isolate;overflow:hidden;min-height:242px;background:linear-gradient(112deg,#0b2535 0%,#123c48 58%,#126d65 100%);border-radius:25px;padding:34px 44px;display:flex;align-items:center;box-shadow:0 18px 36px rgba(20,46,64,.1);}
.hero:before {content:"";position:absolute;right:-56px;top:-155px;width:450px;height:450px;border:1px solid rgba(170,234,218,.20);border-radius:50%;box-shadow:0 0 0 53px rgba(143,228,212,.035),0 0 0 110px rgba(143,228,212,.035);z-index:-1;}
.hero:after {content:"";position:absolute;right:145px;bottom:-195px;width:400px;height:400px;background:radial-gradient(circle,rgba(51,207,173,.23),transparent 69%);z-index:-1;}
.hero-content {position:relative;z-index:1;max-width:660px;}
.hero-eyebrow {display:inline-flex;align-items:center;gap:7px;color:#b9efe5;letter-spacing:.13em;font-weight:800;font-size:10px;text-transform:uppercase;margin-bottom:10px;}
.hero h1 {color:white!important;font-size:clamp(28px,3.5vw,43px)!important;line-height:1.19!important;font-weight:800!important;margin:0 0 13px!important;letter-spacing:-.05em;}
.hero p {color:#d1e6e8;font-size:14px;line-height:1.65;max-width:550px;margin:0 0 18px;}
.hero-pills {display:flex;flex-wrap:wrap;gap:9px;}
.hero-pill {font-size:11px;font-weight:700;border-radius:50px;padding:7px 12px;background:rgba(255,255,255,.11);color:#f2fffb;border:1px solid rgba(255,255,255,.12);}
.hero-icon {position:absolute;right:54px;top:55px;width:170px;height:154px;transform:rotate(11deg);opacity:.92;}
.hero-icon .card-paper {position:absolute;inset:9px 17px;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.22);border-radius:21px;backdrop-filter:blur(6px);transform:rotate(-10deg);}
.hero-icon .mini-line {height:9px;border-radius:8px;background:rgba(207,255,244,.37);position:absolute;left:43px;}
.hero-icon .mini-line.a {top:52px;width:77px;} .hero-icon .mini-line.b{top:75px;width:91px;} .hero-icon .mini-line.c{top:99px;width:51px;background:#72dec0;}
/* Stat strip */
.metrics {display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:19px 0 31px;}
.metric {background:white;border:1px solid #eaf0f5;border-radius:17px;padding:18px 22px;display:flex;align-items:center;gap:13px;box-shadow:0 5px 19px rgba(24,48,68,.035);}
.metric-icon {width:45px;height:45px;border-radius:13px;display:grid;place-items:center;background:#eaf8f3;font-size:21px;flex:none;}
.metric:nth-child(2) .metric-icon {background:#eef1ff;}.metric:nth-child(3) .metric-icon {background:#fff6e9;}
.metric-value {font-family:'Manrope',sans-serif;font-size:20px;font-weight:800;color:#1d2e42;line-height:1.25;}
.metric-label {color:#8290a2;font-weight:600;font-size:11px;margin-top:3px;}
/* Section headers */
.title-row {margin:2px 0 17px;}
.eyebrow {text-transform:uppercase;letter-spacing:.17em;color:#11827c;font-size:10px;font-weight:800;margin-bottom:7px;}
.section-title {font-family:'Manrope',sans-serif;color:#17263d;font-weight:800;font-size:23px;letter-spacing:-.8px;line-height:1.35;}
.section-subtitle {color:#7d8a9b;font-size:12px;line-height:1.6;margin-top:6px;}
.form-heading {display:flex;justify-content:space-between;align-items:center;gap:8px;border-bottom:1px solid #ebeff4;padding:2px 0 13px;margin:14px 0 18px;}
.form-heading h3 {margin:0!important;font-size:14px!important;font-weight:800!important;letter-spacing:-.3px!important;color:#25374a!important;}
.form-heading p {font-size:11px;color:#91a0ad;margin:4px 0 0;}
.form-count {background:#edf7f4;color:#0d8076;font-size:10px;border-radius:30px;padding:6px 9px;font-weight:800;}
/* Streamlit form styling */
[data-testid="stForm"] {background:white;border:1px solid #e7edf3;border-radius:22px;padding:24px 25px 22px;box-shadow:0 10px 29px rgba(18,45,61,.045);}
[data-testid="stForm"] [data-testid="stWidgetLabel"] p {font-size:12px;font-weight:700;color:#35465a;}
[data-testid="stNumberInput"] {width:100%!important;}
[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] [data-baseweb="input"] {width:100%!important;height:44px!important;}
[data-testid="stNumberInput"] [data-baseweb="input"] {border:2px solid #7d8e9d!important;border-radius:10px!important;background:#ffffff!important;box-shadow:none!important;opacity:1!important;}
[data-testid="stNumberInput"] [data-baseweb="input"] > div {height:40px!important;border:none!important;background:#ffffff!important;}
[data-testid="stNumberInput"] input {width:100%!important;height:40px!important;border:none!important;border-radius:8px!important;background:#ffffff!important;color:#17263d!important;-webkit-text-fill-color:#17263d!important;font-size:16px!important;font-weight:800!important;line-height:40px!important;padding:0 12px!important;box-sizing:border-box!important;box-shadow:none!important;opacity:1!important;appearance:textfield;-moz-appearance:textfield;-webkit-appearance:none;}
[data-testid="stNumberInput"] input:focus {outline:none!important;box-shadow:none!important;background:#ffffff!important;color:#17263d!important;-webkit-text-fill-color:#17263d!important;}
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {border-color:#117d77!important;box-shadow:0 0 0 3px rgba(17,125,119,.12)!important;background:#ffffff!important;}
[data-testid="stNumberInput"] button {display:none!important;}
[data-testid="stNumberInput"] input::-webkit-inner-spin-button,
[data-testid="stNumberInput"] input::-webkit-outer-spin-button {appearance:none;-webkit-appearance:none;margin:0;}
[data-testid="stNumberInput"] input[type="number"] {appearance:textfield;-moz-appearance:textfield;}
[data-testid="stFormSubmitButton"] button {background:linear-gradient(90deg,#0f7c73,#119488)!important;border:none!important;border-radius:11px!important;min-height:48px!important;color:white!important;font-size:13px!important;font-weight:800!important;box-shadow:0 8px 20px rgba(19,132,119,.17);}
[data-testid="stFormSubmitButton"] button:hover {filter:brightness(1.05);}
[data-testid="stFormSubmitButton"] {margin-top:10px;}
[data-testid="stButton"] button {border:1px solid #dfe8ed;background:#fff;border-radius:10px;color:#4c6273;font-weight:700;}
/* Result panel */
.result-card {background:white;border:1px solid #e7edf3;border-radius:22px;padding:26px;box-shadow:0 10px 29px rgba(18,45,61,.045);min-height:305px;}
.result-kicker {display:flex;align-items:center;justify-content:space-between;color:#8190a1;text-transform:uppercase;letter-spacing:.1em;font-size:10px;font-weight:800;margin-bottom:22px;}
.result-dot {width:8px;height:8px;border-radius:50%;background:#1ba895;display:inline-block;margin-right:6px;}
.wait-icon {height:100px;width:100px;border-radius:50%;background:radial-gradient(circle,#e9f7f3 33%,#f6fbf9 34%,#f6fbf9 58%,#e5f3ef 59%);display:grid;place-items:center;margin:7px auto 18px;font-size:38px;color:#1e8f84;}
.wait-heading {font-family:Manrope,sans-serif;font-size:18px;font-weight:800;color:#26364b;text-align:center;margin-bottom:9px;}
.wait-text {font-size:12px;color:#8896a6;text-align:center;line-height:1.7;max-width:260px;margin:0 auto 17px;}
.wait-step {display:flex;align-items:center;gap:8px;font-size:11px;color:#6b8190;background:#f6f9fa;border-radius:10px;padding:10px 12px;margin-top:9px;}
.wait-step b {color:#11867c;background:#e5f4ef;border-radius:100px;width:20px;height:20px;display:grid;place-items:center;flex:none;}
.status-label {font-size:10px;letter-spacing:.11em;font-weight:800;text-transform:uppercase;color:#83909e;}
.predicted-label {font-family:Manrope,sans-serif;color:#113c39;font-size:32px;font-weight:800;line-height:1.15;margin:6px 0 10px;}
.fail-label {color:#8c4b28;}
.result-description {font-size:12px;color:#7f8b9b;line-height:1.65;margin-bottom:22px;}
.score-panel {display:flex;align-items:center;gap:18px;background:#f4faf8;padding:18px 16px;border-radius:15px;margin:14px 0;}
.score-panel.fail {background:#fdf7f0;}
.score-ring {width:88px;height:88px;border-radius:50%;display:grid;place-items:center;flex:none;}
.score-core {width:66px;height:66px;background:white;border-radius:50%;display:grid;place-items:center;flex-direction:column;font-size:20px;font-family:Manrope,sans-serif;font-weight:800;color:#1d3b47;}
.score-text-title {font-size:12px;font-weight:800;color:#274453;margin-bottom:4px;}
.score-text-caption {font-size:11px;color:#748995;line-height:1.55;}
.score-line {display:flex;justify-content:space-between;font-size:11px;font-weight:800;color:#536679;margin:17px 0 7px;}
.bar-bg {height:9px;background:#e5ecf0;border-radius:30px;overflow:hidden;}
.bar-fill {height:100%;background:linear-gradient(90deg,#1aa79a,#65c4a5);border-radius:30px;}
.result-note {font-size:11px;color:#7d8f9d;line-height:1.65;margin-top:16px;padding:12px 13px;background:#f8fafb;border:1px solid #e8edf1;border-radius:11px;}
/* Informational cards */
.info-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:8px;}
.info-card {background:white;padding:22px;border:1px solid #e8eef3;border-radius:17px;min-height:155px;}
.info-icon {font-size:23px;line-height:1.3;margin-bottom:12px;}
.info-title {font-family:Manrope,sans-serif;font-size:13px;font-weight:800;color:#223447;margin-bottom:6px;}
.info-desc {font-size:11px;line-height:1.65;color:#7b8b9c;}
.trend-title {font-family:Manrope,sans-serif;font-size:13px;font-weight:800;color:#273d50;margin:3px 0 5px;}
.trend-desc {font-size:11px;color:#8091a1;line-height:1.65;margin-bottom:18px;}
.feature-row {display:flex;justify-content:space-between;gap:10px;margin:13px 0 5px;font-weight:700;font-size:11px;color:#496072;}
.feature-track {height:7px;background:#e9f0f2;border-radius:100px;overflow:hidden;margin-bottom:11px;}
.feature-fill {height:100%;background:linear-gradient(90deg,#38b4a4,#137b78);border-radius:100px;}
.footer-bar {border-top:1px solid #e3eaf0;color:#9aa6b3;font-size:11px;padding:22px 0 0;margin-top:30px;display:flex;align-items:center;justify-content:space-between;gap:10px;}
/* Layout */
[data-testid="stVerticalBlock"] > div:has(> [data-testid="stForm"]) {width:100%;}
@media (max-width:1000px) {.hero-icon{opacity:.25;right:0;}.hero-content{max-width:100%;}}
@media (max-width:700px){
[data-testid="stMainBlockContainer"]{padding:1rem 1rem 4rem;}
.navbar{padding-bottom:14px}.nav-right{font-size:10px}.nav-chip{display:none}
.hero{min-height:0;padding:28px 23px;border-radius:20px;}.hero h1{font-size:29px!important;}.hero p{font-size:12px;}
.hero-icon{right:-80px;opacity:.13}.hero-pill{font-size:10px}
.metrics{gap:8px;margin:12px 0 25px}.metric{padding:12px 10px;gap:6px;flex-direction:column;align-items:flex-start}.metric-icon{width:32px;height:32px;font-size:16px;border-radius:9px}.metric-value{font-size:15px}.metric-label{font-size:9px}
[data-testid="stForm"]{padding:18px 15px;}.result-card{padding:21px;min-height:260px}.info-grid{grid-template-columns:1fr}.info-card{min-height:0;padding:18px}
.footer-bar{flex-direction:column;align-items:flex-start}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading prediction model…")
def load_artifacts():
    """Load the same saved model shipped with the original project."""
    with (BASE / "meta.json").open("r", encoding="utf-8") as f:
        meta_data = json.load(f)
    trained_model = joblib.load(BASE / "model.joblib")
    return trained_model, meta_data


try:
    model, meta = load_artifacts()
except Exception as exc:
    st.error("The saved model could not be loaded. Check that model.joblib and meta.json are beside app.py, and that the required packages are installed.")
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()

FEATURES = meta["features"]
MEDIANS = meta["medians"]
TOP_NAMES = {
    "Submission.Actions": "Submissions",
    "Quiz.Actions": "Quiz activity",
    "Forum.Actions": "Forum interactions",
    "Create.Actions": "Created content",
    "Read.Actions": "Resource views",
    "Update.Actions": "Updates",
    "Delete.Actions": "Deletions",
    "Days.Logged.In": "Days logged in",
}
FIELDS = {
    "Submission.Actions": ("Assignment submissions", "Number of submission actions recorded in the learning platform."),
    "Quiz.Actions": ("Quiz interactions", "Number of quiz-related learning platform actions."),
    "Forum.Actions": ("Forum interactions", "Number of recorded forum activities."),
    "Create.Actions": ("Content created", "Number of create actions in the platform."),
    "Read.Actions": ("Resource views", "Number of read or view actions across course materials."),
    "Update.Actions": ("Update actions", "Number of actions updating platform content."),
    "Delete.Actions": ("Delete actions", "Number of deletion actions in the platform."),
    "Days.Logged.In": ("Days logged in", "Number of days the learner accessed the platform."),
}


def render(html_string: str):
    st.markdown(html_string, unsafe_allow_html=True)


def field(feature: str) -> int:
    label, help_text = FIELDS.get(feature, (feature, "Learning platform activity count."))
    return st.number_input(
        label,
        min_value=0,
        step=1,
        format="%d",
        key=f"feature:{feature}",
        help=help_text,
    )


def reset_inputs(zero: bool = False):
    for feature in FEATURES:
        st.session_state[f"feature:{feature}"] = 0 if zero else int(MEDIANS.get(feature, 0))
    st.session_state.pop("prediction", None)


for name in FEATURES:
    if f"feature:{name}" not in st.session_state:
        st.session_state[f"feature:{name}"] = int(MEDIANS.get(name, 0))

render("""
<div class="navbar">
  <div class="brand"><span class="brand-symbol">✦</span><span>Student<span class="text-muted">Sense</span></span></div>
  <div class="nav-right"><span class="nav-status"></span> AI-assisted academic insights <span class="nav-chip">Research prototype · v2.0</span></div>
</div>
<div class="hero">
  <div class="hero-content">
    <div class="hero-eyebrow">✧ STUDENT SUCCESS INTELLIGENCE</div>
    <h1>Every student journey<br>deserves a clearer picture.</h1>
    <p>Explore learning activity and generate a model-based prediction with a simple, intuitive experience built for educational research.</p>
    <div class="hero-pills"><span class="hero-pill">◉ 8 activity indicators</span><span class="hero-pill">✦ Random Forest model</span><span class="hero-pill">◷ Instant results</span></div>
  </div>
  <div class="hero-icon" aria-hidden="true"><div class="card-paper"></div><div class="mini-line a"></div><div class="mini-line b"></div><div class="mini-line c"></div></div>
</div>
<div class="metrics">
  <div class="metric"><div class="metric-icon">▦</div><div><div class="metric-value">08</div><div class="metric-label">Activity indicators</div></div></div>
  <div class="metric"><div class="metric-icon">✧</div><div><div class="metric-value">Random Forest</div><div class="metric-label">Prediction algorithm</div></div></div>
  <div class="metric"><div class="metric-icon">◎</div><div><div class="metric-value">2 outcomes</div><div class="metric-label">High / low rating</div></div></div>
</div>
""")

left, right = st.columns([1.56, 0.95], gap="large", vertical_alignment="top")
with left:
    render("""
    <div class="title-row"><div class="eyebrow">01 / STUDENT PROFILE</div><div class="section-title">Enter learning activity</div>
    <div class="section-subtitle">Fill in the activity counts from the learning management system, then select Analyze profile.</div></div>
    """)
    with st.form("student_activity", clear_on_submit=False, border=True):
        values = {}
        sections = [
            ("Academic engagement", "Assessment-related actions", ["Submission.Actions", "Quiz.Actions"]),
            ("Learning participation", "Content and platform visits", ["Read.Actions", "Days.Logged.In", "Forum.Actions"]),
            ("Platform interactions", "Additional activity signals", ["Create.Actions", "Update.Actions", "Delete.Actions"]),
        ]
        for title, desc, features in sections:
            available = [f for f in features if f in FEATURES]
            if not available:
                continue
            render(f'<div class="form-heading"><div><h3>{html.escape(title)}</h3><p>{html.escape(desc)}</p></div><span class="form-count">{len(available):02d} fields</span></div>')
            c1, c2 = st.columns(2, gap="medium")
            for ix, name in enumerate(available):
                with (c1 if ix % 2 == 0 else c2):
                    values[name] = field(name)
        # Future-proofed if a supplied meta.json includes unexpected features.
        remaining = [f for f in FEATURES if f not in values]
        if remaining:
            render('<div class="form-heading"><div><h3>Additional indicators</h3><p>Model-specific activity measures</p></div></div>')
            ca, cb = st.columns(2, gap="medium")
            for ix, name in enumerate(remaining):
                with (ca if ix % 2 == 0 else cb):
                    values[name] = field(name)
        st.caption("Counts are non-negative whole numbers. Use actual platform data for meaningful interpretation.")
        submitted = st.form_submit_button("Analyze student profile  →", use_container_width=True, type="primary")
    r1, r2, filler = st.columns([1, 1, 2], gap="small")
    with r1:
        st.button("↻ Reset to typical", key="reset", on_click=reset_inputs, kwargs={"zero": False}, use_container_width=True)
    with r2:
        st.button("Clear all fields", key="clear", on_click=reset_inputs, kwargs={"zero": True}, use_container_width=True)

    if submitted:
        try:
            sample = pd.DataFrame([{f: int(values[f]) for f in FEATURES}], columns=FEATURES)
            class_scores = model.predict_proba(sample)[0]
            classes = list(model.classes_)
            if 1 not in classes or 0 not in classes:
                raise ValueError("This saved model does not contain both expected classes (0 and 1).")
            p_pass = float(class_scores[classes.index(1)])
            score = p_pass if p_pass >= 0.5 else 1.0 - p_pass
            st.session_state["prediction"] = {
                "pass_score": p_pass,
                "class_score": score,
                "predicted_pass": p_pass >= 0.5,
                "values": {f: int(values[f]) for f in FEATURES},
            }
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")

with right:
    render("""
    <div class="title-row"><div class="eyebrow">02 / ANALYSIS</div><div class="section-title">Prediction results</div>
    <div class="section-subtitle">A snapshot of what the trained model estimates from your inputs.</div></div>
    """)
    result = st.session_state.get("prediction")
    if not result:
        render("""
        <div class="result-card">
          <div class="result-kicker"><span>YOUR RESULT APPEARS HERE</span><span><span class="result-dot"></span> Ready</span></div>
          <div class="wait-icon">✧</div>
          <div class="wait-heading">Ready when you are.</div>
          <div class="wait-text">Complete the student activity form to see a predicted rating and an easy-to-read model score.</div>
          <div class="wait-step"><b>1</b> Enter learning activity counts</div>
          <div class="wait-step"><b>2</b> Select Analyze student profile</div>
          <div class="wait-step"><b>3</b> Review the model's estimate</div>
        </div>
        """)
    else:
        pred_pass = bool(result["predicted_pass"])
        pass_score = float(result["pass_score"])
        class_score = float(result["class_score"])
        pass_pct = max(0.0, min(100.0, 100 * pass_score))
        class_pct = max(0.0, min(100.0, 100 * class_score))
        result_label = "Pass (High)" if pred_pass else "Fail (Low)"
        description = (
            "The submitted activity profile was assigned to the dataset's High category."
            if pred_pass else "The submitted activity profile was assigned to the dataset's Low category."
        )
        accent = "#16958a" if pred_pass else "#cd9565"
        render(f"""
        <div class="result-card">
          <div class="result-kicker"><span>MODEL ESTIMATE</span><span><span class="result-dot"></span> Completed</span></div>
          <div class="status-label">Predicted category</div>
          <div class="predicted-label {'fail-label' if not pred_pass else ''}">{html.escape(result_label)}</div>
          <div class="result-description">{html.escape(description)}</div>
          <div class="score-panel {'fail' if not pred_pass else ''}">
             <div class="score-ring" style="background:conic-gradient({accent} 0 {class_pct:.2f}%, #dce8e5 {class_pct:.2f}% 100%)"><div class="score-core">{class_pct:.0f}%</div></div>
             <div><div class="score-text-title">Assigned class score</div><div class="score-text-caption">Model's score for the selected category, not a guaranteed outcome.</div></div>
          </div>
          <div class="score-line"><span>High (Pass) category score</span><span>{pass_pct:.1f}%</span></div>
          <div class="bar-bg"><div class="bar-fill" style="width:{pass_pct:.2f}%"></div></div>
          <div class="result-note"><b>{'Pass indicator' if pred_pass else 'Fail indicator'}:</b> {'Higher learning activity signals are associated with this result.' if pred_pass else 'Lower learning activity signals are associated with this result.'}</div>
        </div>
        """)
        unusual = []
        upper = meta.get("max", {})
        for feature, val in result["values"].items():
            bound = upper.get(feature)
            if bound is not None and val > bound:
                unusual.append(TOP_NAMES.get(feature, feature))
        if unusual:
            st.warning("Some entered values exceed the dataset's 99th-percentile range (" + ", ".join(unusual[:4]) + "). Treat this estimate cautiously.", icon="⚠️")
        st.caption("Results reflect the last submitted profile. Edit the inputs and analyze again to update them.")

    feature_imp = meta.get("importances", {})
    feature_content = ""
    if feature_imp:
        sorted_imp = sorted(feature_imp.items(), key=lambda kv: kv[1], reverse=True)[:4]
        max_val = max(v for _, v in sorted_imp) or 1
        for name, raw in sorted_imp:
            width = min(100, max(0, float(raw) / max_val * 100))
            feature_content += (
                f'<div class="feature-row"><span>{html.escape(TOP_NAMES.get(name, name))}</span>'
                f'<span>{float(raw) * 100:.1f}%</span></div>'
                f'<div class="feature-track"><div class="feature-fill" style="width:{width:.1f}%"></div></div>'
            )
    else:
        feature_content = '<div class="trend-desc">Feature importance data is unavailable for this model.</div>'
    render(
        '<div style="height:14px"></div><div class="result-card" style="min-height:0;padding:23px 26px">'
        '<div class="trend-title">What the model looks at</div>'
        '<div class="trend-desc">Highest global feature importances across the trained Random Forest, not explanations of this particular result.</div>'
        + feature_content + '</div>'
    )

