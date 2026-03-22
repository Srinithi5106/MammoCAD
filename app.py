"""
app.py — MammoCAD: AI-Powered Mammogram Analysis System
Netflix-dark Streamlit UI | Doctor + Lab Assistant roles
Run: streamlit run app.py
"""
import os
import io
import uuid
import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image

# ── Local modules ──────────────────────────────────────────────
from config import APP_TITLE, APP_SUBTITLE, UPLOAD_DIR, REPORTS_DIR
import database as db
import visualizations as viz
from report_generator import generate_report

# ── Page config (must be first Streamlit call) ─────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ─────────────────────────────────────────────────
_css_path = Path("assets/style.css")
if _css_path.exists():
    with open(_css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Extra CSS for auth pages
st.markdown("""
<style>
/* ── General dark background ── */
[data-testid="stAppViewContainer"] {
    background-color: #141414;
}
[data-testid="stSidebar"] {
    background-color: #0a0a0a;
}

/* ── Auth card ── */
.auth-card {
    background: #1f1f1f;
    border: 1px solid #2a2a2a;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-top: 1rem;
}
.mammo-logo {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: 2px;
    color: #FFFFFF;
    margin-bottom: 0;
}
.mammo-logo span { color: #E50914; }

/* ── Tab switcher for login/register ── */
.stTabs [data-baseweb="tab-list"] {
    background: #2a2a2a;
    border-radius: 8px;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #808080;
    font-weight: 600;
    border-radius: 6px;
    padding: 8px 24px;
}
.stTabs [aria-selected="true"] {
    background: #E50914 !important;
    color: #fff !important;
}

/* ── Inputs ── */
input, textarea, select {
    background: #2a2a2a !important;
    color: #fff !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
}
input:focus {
    border-color: #E50914 !important;
    box-shadow: 0 0 0 2px rgba(229,9,20,0.25) !important;
}

/* ── Badges ── */
.badge-malignant {
    background: rgba(229,9,20,0.15);
    color: #E50914;
    border: 1px solid #E50914;
    border-radius: 6px;
    padding: 2px 10px;
    font-weight: 700;
}
.badge-benign {
    background: rgba(70,211,105,0.15);
    color: #46d369;
    border: 1px solid #46d369;
    border-radius: 6px;
    padding: 2px 10px;
    font-weight: 700;
}

/* ── Sidebar nav buttons ── */
section[data-testid="stSidebar"] button {
    border-radius: 8px !important;
    margin-bottom: 4px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Init DB ────────────────────────────────────────────────────
db.init_db()


# ══════════════════════════════════════════════════════════════
# Session state helpers
# ══════════════════════════════════════════════════════════════

def _init_session():
    defaults = {
        "logged_in":  False,
        "user":       None,
        "page":       "login",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _logout():
    st.session_state.logged_in = False
    st.session_state.user      = None
    st.session_state.page      = "login"
    st.rerun()


# ══════════════════════════════════════════════════════════════
# Shared UI helpers
# ══════════════════════════════════════════════════════════════

def _logo():
    st.markdown(
        '<div class="mammo-logo">MAMMO<span>CAD</span></div>',
        unsafe_allow_html=True
    )


def _sidebar_nav(role: str):
    with st.sidebar:
        _logo()
        st.markdown("---")
        user = st.session_state.user
        st.markdown(
            f"**{user['full_name']}**  \n"
            f"<span style='color:#808080;font-size:0.8rem'>"
            f"{'🩺 Doctor' if role=='doctor' else '🔬 Lab Assistant'}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        if role == "lab_assistant":
            pages = {
                "📤 Upload & Analyse":  "upload",
                "👤 My Patients":       "my_patients",
            }
        else:
            pages = {
                "📊 Overview":          "overview",
                "👥 All Patients":      "all_patients",
                "🔬 Analysis Details":  "analysis_detail",
            }

        for label, key in pages.items():
            active = st.session_state.page == key
            if st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.page = key
                st.rerun()

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            _logout()


# ══════════════════════════════════════════════════════════════
# Login / Register page
# ══════════════════════════════════════════════════════════════

def page_login():
    col_l, col_m, col_r = st.columns([1, 1.4, 1])
    with col_m:
        st.markdown("<br><br>", unsafe_allow_html=True)
        _logo()
        st.markdown(
            f"<p style='color:#808080;font-size:1rem;margin-top:-6px'>"
            f"{APP_SUBTITLE}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔐 Sign In", "📝 Register"])

        # ── SIGN IN TAB ────────────────────────────────────────
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)

            role_display = st.radio(
                "Sign in as",
                ["🩺 Doctor", "🔬 Lab Assistant"],
                horizontal=True,
                key="login_role",
            )
            role = "doctor" if "Doctor" in role_display else "lab_assistant"

            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter username", key="login_user")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Sign In", use_container_width=True, key="btn_signin"):
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    user = db.verify_user(username, password)
                    if user and user["role"] == role:
                        st.session_state.logged_in = True
                        st.session_state.user      = user
                        st.session_state.page      = "upload" if role == "lab_assistant" else "overview"
                        st.rerun()
                    elif user and user["role"] != role:
                        st.error(
                            f"⚠️ This account is registered as "
                            f"**{'Doctor' if user['role']=='doctor' else 'Lab Assistant'}**, "
                            f"not **{'Doctor' if role=='doctor' else 'Lab Assistant'}**. "
                            f"Please select the correct role."
                        )
                    else:
                        st.error("❌ Invalid username or password.")

            st.markdown(
                "<p style='color:#444;font-size:0.75rem;text-align:center;margin-top:1rem'>"
                "Default accounts: <b style='color:#666'>doctor1 / doc123</b> &nbsp;·&nbsp; "
                "<b style='color:#666'>labtech1 / lab123</b></p>",
                unsafe_allow_html=True,
            )

        # ── REGISTER TAB ───────────────────────────────────────
        with tab_register:
            st.markdown("<br>", unsafe_allow_html=True)

            reg_role_display = st.radio(
                "Registering as",
                ["🩺 Doctor", "🔬 Lab Assistant"],
                horizontal=True,
                key="reg_role",
            )
            reg_role = "doctor" if "Doctor" in reg_role_display else "lab_assistant"

            reg_name     = st.text_input("Full Name *", placeholder="Dr. Jane Smith", key="reg_name")
            reg_username = st.text_input("Username *", placeholder="Choose a username", key="reg_username")

            c1, c2 = st.columns(2)
            reg_password  = c1.text_input("Password *", type="password", placeholder="Min 6 chars", key="reg_pass")
            reg_password2 = c2.text_input("Confirm Password *", type="password", placeholder="Repeat password", key="reg_pass2")

            reg_email = st.text_input("Email (optional)", placeholder="your@hospital.com", key="reg_email")

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Create Account", use_container_width=True, key="btn_register"):
                # Validation
                if not reg_name or not reg_username or not reg_password or not reg_password2:
                    st.error("❌ Please fill in all required fields (marked with *).")
                elif len(reg_username) < 3:
                    st.error("❌ Username must be at least 3 characters.")
                elif len(reg_password) < 6:
                    st.error("❌ Password must be at least 6 characters.")
                elif reg_password != reg_password2:
                    st.error("❌ Passwords do not match.")
                elif not reg_name.strip():
                    st.error("❌ Please enter your full name.")
                else:
                    ok, msg = db.register_user(
                        username  = reg_username.strip(),
                        password  = reg_password,
                        role      = reg_role,
                        full_name = reg_name.strip(),
                        email     = reg_email.strip() if reg_email else "",
                    )
                    if ok:
                        st.success(
                            f"✅ Account created for **{reg_name}**! "
                            f"Switch to the Sign In tab to log in."
                        )
                        st.balloons()
                    else:
                        st.error(f"❌ {msg}")

            st.markdown(
                "<p style='color:#444;font-size:0.75rem;text-align:center;margin-top:1rem'>"
                "Accounts require admin approval in production deployments.</p>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════
# Lab Assistant — Upload & Analyse
# ══════════════════════════════════════════════════════════════

def page_upload():
    st.title("📤 Upload & Analyse Mammogram")
    st.markdown("---")

    col1, col2 = st.columns([1, 1.6], gap="large")

    # ── Left: Patient form ─────────────────────────────────────
    with col1:
        st.subheader("Patient Details")
        with st.form("patient_form", clear_on_submit=False):
            patient_id = st.text_input(
                "Patient ID *",
                placeholder="e.g. PT-2024-001",
                help="Unique identifier — used to link all records."
            )
            full_name  = st.text_input("Full Name *", placeholder="Patient's full name")
            c1, c2     = st.columns(2)
            age        = c1.number_input("Age", min_value=1, max_value=120, value=45)
            contact    = c2.text_input("Contact", placeholder="Phone / Email")
            history    = st.text_area(
                "Clinical History",
                placeholder="Family history, prior biopsies, symptoms…",
                height=100,
            )

            st.markdown("---")
            st.subheader("Upload Mammogram")
            uploaded_file = st.file_uploader(
                "Drop mammogram image here",
                type=["png", "jpg", "jpeg", "dcm"],
                help="Accepts PNG, JPEG. DICOM support coming soon.",
            )
            notes = st.text_area("Radiologist Notes", height=80,
                                 placeholder="Any observations before AI analysis…")
            submitted = st.form_submit_button("🔬 Analyse", use_container_width=True)

        if submitted:
            if not patient_id or not full_name:
                st.error("Patient ID and Full Name are required.")
                return
            if not uploaded_file:
                st.error("Please upload a mammogram image.")
                return

            # Save patient
            ok, msg = db.add_patient(
                patient_id, full_name, age, contact, history,
                st.session_state.user["username"]
            )
            if not ok and "exists" not in msg:
                st.error(msg)
                return

            # Save image
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            ext       = Path(uploaded_file.name).suffix or ".png"
            fname     = f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            img_path  = os.path.join(UPLOAD_DIR, fname)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Run AI prediction
            with st.spinner("Running AI analysis…"):
                try:
                    from predict import predict_image
                    result = predict_image(img_path)
                except FileNotFoundError as e:
                    st.error(str(e))
                    return
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    return

            # Save to DB
            db.save_analysis(
                patient_id     = patient_id,
                image_path     = img_path,
                prediction     = result["prediction"],
                benign_prob    = result["benign_prob"],
                malignant_prob = result["malignant_prob"],
                birads_category= result["birads_category"],
                birads_desc    = result["birads_desc"],
                features_dict  = result["features"],
                notes          = notes,
                analysed_by    = st.session_state.user["username"],
            )

            st.session_state["last_result"]      = result
            st.session_state["last_img_path"]    = img_path
            st.session_state["last_patient_id"]  = patient_id
            st.session_state["last_patient_name"]= full_name
            st.rerun()

    # ── Right: Results ─────────────────────────────────────────
    with col2:
        if "last_result" not in st.session_state:
            st.markdown(
                "<div style='display:flex;align-items:center;justify-content:center;"
                "height:400px;color:#333;font-size:1.1rem;border:2px dashed #222;"
                "border-radius:12px'>"
                "Analysis results will appear here after upload"
                "</div>",
                unsafe_allow_html=True,
            )
            return

        result      = st.session_state["last_result"]
        img_path    = st.session_state["last_img_path"]
        patient_name= st.session_state.get("last_patient_name", "")

        pred = result["prediction"]
        badge_class = "badge-malignant" if pred == "Malignant" else "badge-benign"
        st.markdown(
            f"<h2>Result: <span class='{badge_class}'>{pred.upper()}</span></h2>",
            unsafe_allow_html=True,
        )

        r1, r2 = st.columns(2)
        r1.metric("Benign Probability",    f"{result['benign_prob']*100:.1f}%")
        r2.metric("Malignant Probability", f"{result['malignant_prob']*100:.1f}%")

        st.markdown(
            f"**{result['birads_category']}** — {result['birads_desc']}",
        )

        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            st.image(img, caption="Uploaded Mammogram", width=700)

        st.plotly_chart(
            viz.probability_gauge(result["benign_prob"], result["malignant_prob"]),
            use_container_width=True, key="gauge_upload"
        )
        st.plotly_chart(
            viz.probability_bar(result["benign_prob"], result["malignant_prob"]),
            use_container_width=True, key="prob_bar_upload"
        )
        st.plotly_chart(
            viz.birads_distribution_chart(result["birads_category"]),
            use_container_width=True, key="birads_upload"
        )

        st.markdown("---")
        st.subheader("Cell Feature Profile")
        st.plotly_chart(
            viz.radar_chart(result["features"], patient_name),
            use_container_width=True, key="radar_upload"
        )
        st.plotly_chart(
            viz.feature_bar_chart(result["features"]),
            use_container_width=True, key="feat_bar_upload"
        )


# ══════════════════════════════════════════════════════════════
# Lab Assistant — My Patients
# ══════════════════════════════════════════════════════════════

def page_my_patients():
    st.title("👤 My Patients")
    st.markdown("---")

    search = st.text_input("🔍 Search by name or ID", placeholder="Type to search…")
    if search:
        patients = db.search_patients(search)
    else:
        patients = db.get_all_patients()

    if not patients:
        st.info("No patients found.")
        return

    st.markdown(f"**{len(patients)} patient(s) found**")

    for p in patients:
        with st.expander(f"🧑 {p['full_name']}  —  ID: {p['patient_id']}  (Age: {p['age']})", expanded=False):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Contact:** {p.get('contact','—')}")
            c2.markdown(f"**Registered:** {p.get('created_at','—')[:10]}")
            if p.get("history"):
                st.markdown(f"**History:** {p['history']}")

            analyses = db.get_analyses_for_patient(p["patient_id"])
            if analyses:
                st.markdown(f"**{len(analyses)} analysis record(s):**")
                for a in analyses:
                    pred   = a.get("prediction","—")
                    color  = "#E50914" if pred == "Malignant" else "#46d369"
                    st.markdown(
                        f"&nbsp;&nbsp;• `{a['analysed_at'][:16]}` — "
                        f"<span style='color:{color};font-weight:700'>{pred}</span> "
                        f"| {a.get('birads_category','—')} "
                        f"| By: {a.get('analysed_by','—')}",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No analyses yet.")


# ══════════════════════════════════════════════════════════════
# Doctor — Overview
# ══════════════════════════════════════════════════════════════

def page_overview():
    st.title("📊 Dashboard Overview")
    st.markdown("---")

    stats = db.get_stats()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Patients",  stats["total_patients"])
    m2.metric("Total Analyses",  stats["total_analyses"])
    m3.metric("Malignant Cases", stats["malignant_count"])
    m4.metric("Benign Cases",    stats["benign_count"])

    all_analyses = db.get_all_analyses()

    if not all_analyses:
        st.info("No analyses have been performed yet.")
        return

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            viz.population_pie(stats["benign_count"], stats["malignant_count"]),
            use_container_width=True, key="overview_pie"
        )
    with c2:
        st.plotly_chart(
            viz.birads_histogram(all_analyses),
            use_container_width=True, key="overview_birads"
        )

    st.plotly_chart(
        viz.analyses_over_time(all_analyses),
        use_container_width=True, key="overview_timeline"
    )
    st.plotly_chart(
        viz.feature_scatter_matrix(all_analyses),
        use_container_width=True, key="overview_scatter"
    )

    st.markdown("---")
    st.subheader("Recent Analyses")
    rows = []
    for a in all_analyses[:20]:
        rows.append({
            "Patient ID":   a["patient_id"],
            "Name":         a.get("full_name", "—"),
            "Age":          a.get("age", "—"),
            "Prediction":   a.get("prediction","—"),
            "BI-RADS":      a.get("birads_category","—"),
            "Malignant %":  f"{a.get('malignant_prob',0)*100:.1f}%",
            "Date":         a.get("analysed_at","")[:16],
            "By":           a.get("analysed_by","—"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# Doctor — All Patients
# ══════════════════════════════════════════════════════════════

def page_all_patients():
    st.title("👥 All Patient Records")
    st.markdown("---")

    search = st.text_input("🔍 Search by name or ID")
    patients = db.search_patients(search) if search else db.get_all_patients()

    if not patients:
        st.info("No patients found.")
        return

    st.markdown(f"**{len(patients)} patient(s)**")

    for p in patients:
        pid = p["patient_id"]
        with st.expander(
            f"🧑 **{p['full_name']}**  ·  {pid}  ·  Age {p['age']}",
            expanded=False,
        ):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Contact:** {p.get('contact','—')}")
            c2.markdown(f"**Registered:** {p.get('created_at','—')[:10]}")
            c3.markdown(f"**By:** {p.get('created_by','—')}")
            if p.get("history"):
                st.markdown(f"**Clinical History:** {p['history']}")

            analyses = db.get_analyses_for_patient(pid)
            if not analyses:
                st.caption("No analyses for this patient.")
                continue

            for idx, a in enumerate(analyses):
                pred   = a.get("prediction","—")
                color  = "#E50914" if pred == "Malignant" else "#46d369"
                with st.container():
                    st.markdown(
                        f"<div style='border-left:3px solid {color};"
                        f"padding-left:12px;margin:8px 0'>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**Analysis #{idx+1}** — `{a['analysed_at'][:16]}`  "
                        f"<span style='color:{color};font-weight:700'>{pred}</span>  "
                        f"| {a.get('birads_category','—')} "
                        f"| Malignant: {a.get('malignant_prob',0)*100:.1f}%",
                        unsafe_allow_html=True,
                    )
                    if a.get("features"):
                        tc1, tc2 = st.columns(2)
                        with tc1:
                            st.plotly_chart(
                                viz.radar_chart(a["features"], p["full_name"]),
                                use_container_width=True,
                                key=f"radar_{pid}_{idx}"
                            )
                        with tc2:
                            st.plotly_chart(
                                viz.probability_gauge(
                                    a.get("benign_prob", 0),
                                    a.get("malignant_prob", 0),
                                ),
                                use_container_width=True,
                                key=f"gauge_{pid}_{idx}"
                            )
                        st.plotly_chart(
                            viz.feature_bar_chart(a["features"]),
                            use_container_width=True,
                            key=f"fbar_{pid}_{idx}"
                        )

                    img_path = a.get("image_path","")
                    if img_path and os.path.exists(img_path):
                        st.image(
                            Image.open(img_path).convert("RGB"),
                            caption="Mammogram", width=300,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                    if st.button(
                        f"📄 Download Report — Analysis #{idx+1}",
                        key=f"rep_{pid}_{idx}"
                    ):
                        _generate_and_download_report(p, a)

            st.markdown("---")


# ══════════════════════════════════════════════════════════════
# Doctor — Analysis Detail
# ══════════════════════════════════════════════════════════════

def page_analysis_detail():
    st.title("🔬 Analysis Detail")
    st.markdown("---")

    all_patients = db.get_all_patients()
    if not all_patients:
        st.info("No patients found.")
        return

    pid_options = [f"{p['patient_id']} — {p['full_name']}" for p in all_patients]
    selected    = st.selectbox("Select Patient", pid_options)
    pid         = selected.split(" — ")[0]
    patient     = db.get_patient(pid)
    analyses    = db.get_analyses_for_patient(pid)

    if not analyses:
        st.info("No analyses for this patient.")
        return

    dates  = [f"#{i+1}: {a['analysed_at'][:16]} — {a['prediction']}"
              for i, a in enumerate(analyses)]
    choice = st.selectbox("Select Analysis", dates)
    idx    = dates.index(choice)
    a      = analyses[idx]

    st.markdown("---")

    pred  = a.get("prediction","—")
    badge = "badge-malignant" if pred == "Malignant" else "badge-benign"
    st.markdown(
        f"<h2>Diagnosis: <span class='{badge}'>{pred.upper()}</span></h2>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Benign",    f"{a.get('benign_prob',0)*100:.1f}%")
    c2.metric("Malignant", f"{a.get('malignant_prob',0)*100:.1f}%")
    c3.metric("BI-RADS",   a.get("birads_category","—"))
    c4.metric("Date",      a.get("analysed_at","—")[:10])

    st.caption(f"**Assessment:** {a.get('birads_desc','—')}")
    if a.get("notes"):
        st.info(f"📝 Notes: {a['notes']}")

    st.markdown("---")

    t1, t2, t3 = st.tabs(["📈 Probability", "🕸 Cell Profile", "📊 Features"])

    with t1:
        cc1, cc2 = st.columns(2)
        cc1.plotly_chart(
            viz.probability_gauge(a.get("benign_prob",0), a.get("malignant_prob",0)),
            use_container_width=True, key="det_gauge"
        )
        cc2.plotly_chart(
            viz.probability_bar(a.get("benign_prob",0), a.get("malignant_prob",0)),
            use_container_width=True, key="det_pbar"
        )
        st.plotly_chart(
            viz.birads_distribution_chart(a.get("birads_category","")),
            use_container_width=True, key="det_birads"
        )

    with t2:
        if a.get("features"):
            st.plotly_chart(
                viz.radar_chart(a["features"], patient["full_name"]),
                use_container_width=True, key="det_radar"
            )
        else:
            st.info("No feature data available.")

    with t3:
        if a.get("features"):
            st.plotly_chart(
                viz.feature_bar_chart(a["features"]),
                use_container_width=True, key="det_fbar"
            )
            feats   = a["features"]
            mean_kk = sorted([k for k in feats if k.endswith("_mean")])
            rows    = []
            for k in mean_kk:
                wk = k.replace("_mean","_worst")
                sk = k.replace("_mean","_se")
                rows.append({
                    "Feature":  k.replace("_mean","").replace("_"," ").title(),
                    "Mean":     round(feats[k], 4),
                    "Worst":    round(feats.get(wk, 0), 4),
                    "SE":       round(feats.get(sk, 0), 4),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    img_path = a.get("image_path","")
    if img_path and os.path.exists(img_path):
        st.markdown("---")
        st.subheader("Mammogram Image")
        img = Image.open(img_path).convert("RGB")
        col_img, _ = st.columns([1, 1])
        col_img.image(img, caption="Uploaded Mammogram", width=600)

    st.markdown("---")
    if st.button("📄 Generate & Download PDF Report", type="primary"):
        _generate_and_download_report(patient, a)


# ══════════════════════════════════════════════════════════════
# Report helper
# ══════════════════════════════════════════════════════════════

def _generate_and_download_report(patient: dict, analysis: dict):
    with st.spinner("Generating PDF report…"):
        try:
            figs = {
                "Probability Gauge": viz.probability_gauge(
                    analysis.get("benign_prob",0),
                    analysis.get("malignant_prob",0),
                ),
                "Cell Feature Profile": viz.radar_chart(
                    analysis.get("features",{}),
                    patient.get("full_name",""),
                ),
                "Feature Comparison": viz.feature_bar_chart(
                    analysis.get("features",{}),
                ),
            }
        except Exception:
            figs = {}

        try:
            path = generate_report(patient, analysis, figs)
        except Exception as e:
            st.error(f"Report generation failed: {e}")
            return

    with open(path, "rb") as f:
        st.download_button(
            label="⬇️ Download PDF",
            data=f.read(),
            file_name=Path(path).name,
            mime="application/pdf",
            key=f"dl_{uuid.uuid4().hex[:6]}",
        )
    st.success("Report ready! Click ⬇️ to download.")


# ══════════════════════════════════════════════════════════════
# Router
# ══════════════════════════════════════════════════════════════

def main():
    _init_session()

    if not st.session_state.logged_in:
        page_login()
        return

    user = st.session_state.user
    role = user["role"]

    _sidebar_nav(role)

    page = st.session_state.page

    if role == "lab_assistant":
        if page == "upload":
            page_upload()
        elif page == "my_patients":
            page_my_patients()
        else:
            page_upload()

    elif role == "doctor":
        if page == "overview":
            page_overview()
        elif page == "all_patients":
            page_all_patients()
        elif page == "analysis_detail":
            page_analysis_detail()
        else:
            page_overview()


if __name__ == "__main__":
    main()