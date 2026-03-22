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

from config import APP_TITLE, APP_SUBTITLE, UPLOAD_DIR, REPORTS_DIR
import database as db
import visualizations as viz
from report_generator import generate_report

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else ":material/radiology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css_path = Path("assets/style.css")
if _css_path.exists():
    with open(_css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
/* Font Awesome */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css');

[data-testid="stAppViewContainer"] { background-color: #141414; }
[data-testid="stSidebar"]          { background-color: #0a0a0a; }

.mammo-logo { font-size:2.4rem; font-weight:900; letter-spacing:2px; color:#FFFFFF; margin-bottom:0; }
.mammo-logo span { color:#E50914; }

/* Nav item icons */
.nav-icon { margin-right: 8px; width: 16px; text-align: center; }

/* Sidebar buttons */
section[data-testid="stSidebar"] button {
    border-radius: 8px !important;
    margin-bottom: 4px !important;
    font-weight: 600 !important;
    text-align: left !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:#2a2a2a; border-radius:8px; gap:0; }
.stTabs [data-baseweb="tab"]      { color:#808080; font-weight:600; border-radius:6px; padding:8px 24px; }
.stTabs [aria-selected="true"]    { background:#E50914 !important; color:#fff !important; }

/* Inputs */
input, textarea, select {
    background:#2a2a2a !important; color:#fff !important;
    border:1px solid #333 !important; border-radius:8px !important;
}
input:focus { border-color:#E50914 !important; box-shadow:0 0 0 2px rgba(229,9,20,0.25) !important; }

/* Badges */
.badge-malignant {
    background:rgba(229,9,20,0.15); color:#E50914;
    border:1px solid #E50914; border-radius:6px; padding:2px 10px; font-weight:700;
}
.badge-benign {
    background:rgba(70,211,105,0.15); color:#46d369;
    border:1px solid #46d369; border-radius:6px; padding:2px 10px; font-weight:700;
}

/* FA icon helper classes */
.fa-red   { color: #E50914; }
.fa-green { color: #46d369; }
.fa-gray  { color: #808080; }

/* User info block in sidebar */
.user-info { padding: 8px 4px; line-height: 1.6; }
.user-role { color:#808080; font-size:0.82rem; }
</style>
""", unsafe_allow_html=True)

db.init_db()


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _init_session():
    defaults = {"logged_in": False, "user": None, "page": "login"}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _logout():
    st.session_state.logged_in = False
    st.session_state.user      = None
    st.session_state.page      = "login"
    st.rerun()


def _fa(icon: str, classes: str = "", style: str = "") -> str:
    """Return an inline Font Awesome icon HTML tag."""
    return f'<i class="fa-solid {icon} {classes}" style="{style}"></i>'


def _logo():
    st.markdown('<div class="mammo-logo">MAMMO<span>CAD</span></div>', unsafe_allow_html=True)


def _sidebar_nav(role: str):
    with st.sidebar:
        _logo()
        st.markdown("---")

        user = st.session_state.user
        role_label = "Doctor" if role == "doctor" else "Lab Assistant"
        role_icon  = "fa-user-doctor" if role == "doctor" else "fa-flask"
        st.markdown(
            f"<div class='user-info'>"
            f"<strong>{user['full_name']}</strong><br>"
            f"<span class='user-role'>{_fa(role_icon)} {role_label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        if role == "lab_assistant":
            pages = {
                "upload":      ("fa-upload",       "Upload & Analyse"),
                "my_patients": ("fa-user",          "My Patients"),
            }
        else:
            pages = {
                "overview":        ("fa-chart-pie",    "Overview"),
                "all_patients":    ("fa-users",         "All Patients"),
                "analysis_detail": ("fa-microscope",    "Analysis Details"),
            }

        for key, (icon, label) in pages.items():
            active = st.session_state.page == key
            btn_label = f"{label}"
            if st.button(
                btn_label,
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.page = key
                st.rerun()

        st.markdown("---")
        if st.button("Logout", key="btn_logout", use_container_width=True, type="secondary"):
            _logout()


# ══════════════════════════════════════════════════════════════
# Login / Register
# ══════════════════════════════════════════════════════════════

def page_login():
    col_l, col_m, col_r = st.columns([1, 1.4, 1])
    with col_m:
        st.markdown("<br><br>", unsafe_allow_html=True)
        _logo()
        st.markdown(
            f"<p style='color:#808080;font-size:1rem;margin-top:-6px'>{APP_SUBTITLE}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["Sign In", "Register"])

        # ── SIGN IN ────────────────────────────────────────────
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            role_display = st.radio(
                "Sign in as",
                ["Doctor", "Lab Assistant"],
                horizontal=True,
                key="login_role",
                captions=None,
            )
            role = "doctor" if role_display == "Doctor" else "lab_assistant"

            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", placeholder="Enter username", key="login_user")
            password = st.text_input("Password", type="password", placeholder="Password", key="login_pass")
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
                        correct = "Doctor" if user["role"] == "doctor" else "Lab Assistant"
                        st.error(f"This account is registered as {correct}. Please select the correct role.")
                    else:
                        st.error("Invalid username or password.")

            st.markdown(
                "<p style='color:#444;font-size:0.75rem;text-align:center;margin-top:1rem'>"
                "Default accounts: doctor1 / doc123 &nbsp;·&nbsp; labtech1 / lab123</p>",
                unsafe_allow_html=True,
            )

        # ── REGISTER ───────────────────────────────────────────
        with tab_register:
            st.markdown("<br>", unsafe_allow_html=True)
            reg_role_display = st.radio(
                "Registering as",
                ["Doctor", "Lab Assistant"],
                horizontal=True,
                key="reg_role",
            )
            reg_role = "doctor" if reg_role_display == "Doctor" else "lab_assistant"

            reg_name     = st.text_input("Full Name *",     placeholder="Dr. Jane Smith",  key="reg_name")
            reg_username = st.text_input("Username *",      placeholder="Choose a username", key="reg_username")
            c1, c2 = st.columns(2)
            reg_password  = c1.text_input("Password *",         type="password", placeholder="Min 6 chars", key="reg_pass")
            reg_password2 = c2.text_input("Confirm Password *", type="password", placeholder="Repeat",       key="reg_pass2")
            reg_email = st.text_input("Email (optional)", placeholder="your@hospital.com", key="reg_email")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Create Account", use_container_width=True, key="btn_register"):
                if not reg_name or not reg_username or not reg_password or not reg_password2:
                    st.error("Please fill in all required fields.")
                elif len(reg_username) < 3:
                    st.error("Username must be at least 3 characters.")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters.")
                elif reg_password != reg_password2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = db.register_user(
                        username  = reg_username.strip(),
                        password  = reg_password,
                        role      = reg_role,
                        full_name = reg_name.strip(),
                        email     = reg_email.strip() if reg_email else "",
                    )
                    if ok:
                        st.success(f"Account created for {reg_name}. Switch to Sign In to log in.")
                        st.balloons()
                    else:
                        st.error(msg)

            st.markdown(
                "<p style='color:#444;font-size:0.75rem;text-align:center;margin-top:1rem'>"
                "Accounts require admin approval in production deployments.</p>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════
# Lab Assistant — Upload & Analyse
# ══════════════════════════════════════════════════════════════

def page_upload():
    st.markdown(
        f"<h1>{_fa('fa-upload', 'fa-red')} &nbsp;Upload &amp; Analyse Mammogram</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 1.6], gap="large")

    with col1:
        st.markdown(
            f"<h3>{_fa('fa-id-card')} &nbsp;Patient Details</h3>",
            unsafe_allow_html=True,
        )
        with st.form("patient_form", clear_on_submit=False):
            patient_id = st.text_input("Patient ID *", placeholder="e.g. PT-2024-001")
            full_name  = st.text_input("Full Name *",  placeholder="Patient's full name")
            c1, c2 = st.columns(2)
            age     = c1.number_input("Age", min_value=1, max_value=120, value=45)
            contact = c2.text_input("Contact", placeholder="Phone / Email")
            history = st.text_area("Clinical History",
                                   placeholder="Family history, prior biopsies, symptoms...",
                                   height=100)
            st.markdown("---")
            st.markdown(
                f"<h4>{_fa('fa-image')} &nbsp;Upload Mammogram</h4>",
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Drop mammogram image here",
                type=["png", "jpg", "jpeg", "dcm"],
            )
            notes     = st.text_area("Radiologist Notes", height=80,
                                     placeholder="Any observations before AI analysis...")
            submitted = st.form_submit_button("Run Analysis", use_container_width=True)

        if submitted:
            if not patient_id or not full_name:
                st.error("Patient ID and Full Name are required.")
                return
            if not uploaded_file:
                st.error("Please upload a mammogram image.")
                return

            ok, msg = db.add_patient(
                patient_id, full_name, age, contact, history,
                st.session_state.user["username"]
            )
            if not ok and "exists" not in msg:
                st.error(msg)
                return

            os.makedirs(UPLOAD_DIR, exist_ok=True)
            ext      = Path(uploaded_file.name).suffix or ".png"
            fname    = f"{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            img_path = os.path.join(UPLOAD_DIR, fname)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Running AI analysis..."):
                try:
                    from predict import predict_image
                    result = predict_image(img_path)
                except FileNotFoundError as e:
                    st.error(str(e))
                    return
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    return

            db.save_analysis(
                patient_id      = patient_id,
                image_path      = img_path,
                prediction      = result["prediction"],
                benign_prob     = result["benign_prob"],
                malignant_prob  = result["malignant_prob"],
                birads_category = result["birads_category"],
                birads_desc     = result["birads_desc"],
                features_dict   = result["features"],
                notes           = notes,
                analysed_by     = st.session_state.user["username"],
            )

            st.session_state["last_result"]       = result
            st.session_state["last_img_path"]     = img_path
            st.session_state["last_patient_id"]   = patient_id
            st.session_state["last_patient_name"] = full_name
            st.rerun()

    with col2:
        if "last_result" not in st.session_state:
            st.markdown(
                "<div style='display:flex;align-items:center;justify-content:center;"
                "height:400px;color:#333;font-size:1.1rem;border:2px dashed #222;"
                "border-radius:12px'>Analysis results will appear here after upload</div>",
                unsafe_allow_html=True,
            )
            return

        result       = st.session_state["last_result"]
        img_path     = st.session_state["last_img_path"]
        patient_name = st.session_state.get("last_patient_name", "")
        pred         = result["prediction"]
        badge_class  = "badge-malignant" if pred == "Malignant" else "badge-benign"

        st.markdown(
            f"<h2>{_fa('fa-circle-check')} &nbsp;Result: "
            f"<span class='{badge_class}'>{pred.upper()}</span></h2>",
            unsafe_allow_html=True,
        )

        r1, r2 = st.columns(2)
        r1.metric("Benign Probability",    f"{result['benign_prob']*100:.1f}%")
        r2.metric("Malignant Probability", f"{result['malignant_prob']*100:.1f}%")

        st.markdown(f"**{result['birads_category']}** - {result['birads_desc']}")

        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            st.image(img, caption="Uploaded Mammogram", width=700)

        st.plotly_chart(viz.probability_gauge(result["benign_prob"], result["malignant_prob"]),
                        use_container_width=True, key="gauge_upload")
        st.plotly_chart(viz.probability_bar(result["benign_prob"], result["malignant_prob"]),
                        use_container_width=True, key="prob_bar_upload")
        st.plotly_chart(viz.birads_distribution_chart(result["birads_category"]),
                        use_container_width=True, key="birads_upload")

        st.markdown("---")
        st.markdown(
            f"<h3>{_fa('fa-dna')} &nbsp;Cell Feature Profile</h3>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(viz.radar_chart(result["features"], patient_name),
                        use_container_width=True, key="radar_upload")
        st.plotly_chart(viz.feature_bar_chart(result["features"]),
                        use_container_width=True, key="feat_bar_upload")


# ══════════════════════════════════════════════════════════════
# Lab Assistant — My Patients
# ══════════════════════════════════════════════════════════════

def page_my_patients():
    st.markdown(
        f"<h1>{_fa('fa-user', 'fa-red')} &nbsp;My Patients</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    search = st.text_input("Search by name or ID", placeholder="Type to search...")
    patients = db.search_patients(search) if search else db.get_all_patients()

    if not patients:
        st.info("No patients found.")
        return

    st.markdown(f"**{len(patients)} patient(s) found**")

    for p in patients:
        label = (
            f"{_fa('fa-user-injured')} &nbsp;"
            f"{p['full_name']}  —  ID: {p['patient_id']}  (Age: {p['age']})"
        )
        with st.expander(p['full_name'] + "  |  " + p['patient_id'], expanded=False):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Contact:** {p.get('contact', '-')}")
            c2.markdown(f"**Registered:** {p.get('created_at', '-')[:10]}")
            if p.get("history"):
                st.markdown(f"**History:** {p['history']}")

            analyses = db.get_analyses_for_patient(p["patient_id"])
            if analyses:
                st.markdown(f"**{len(analyses)} analysis record(s):**")
                for a in analyses:
                    pred  = a.get("prediction", "-")
                    color = "#E50914" if pred == "Malignant" else "#46d369"
                    icon  = "fa-triangle-exclamation" if pred == "Malignant" else "fa-circle-check"
                    st.markdown(
                        f"&nbsp;&nbsp;{_fa(icon, style=f'color:{color}')} "
                        f"`{a['analysed_at'][:16]}` &nbsp;|&nbsp; "
                        f"<span style='color:{color};font-weight:700'>{pred}</span> "
                        f"&nbsp;|&nbsp; {a.get('birads_category', '-')} "
                        f"&nbsp;|&nbsp; By: {a.get('analysed_by', '-')}",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No analyses yet.")


# ══════════════════════════════════════════════════════════════
# Doctor — Overview
# ══════════════════════════════════════════════════════════════

def page_overview():
    st.markdown(
        f"<h1>{_fa('fa-chart-pie', 'fa-red')} &nbsp;Dashboard Overview</h1>",
        unsafe_allow_html=True,
    )
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
        st.plotly_chart(viz.population_pie(stats["benign_count"], stats["malignant_count"]),
                        use_container_width=True, key="overview_pie")
    with c2:
        st.plotly_chart(viz.birads_histogram(all_analyses),
                        use_container_width=True, key="overview_birads")

    st.plotly_chart(viz.analyses_over_time(all_analyses),
                    use_container_width=True, key="overview_timeline")
    st.plotly_chart(viz.feature_scatter_matrix(all_analyses),
                    use_container_width=True, key="overview_scatter")

    st.markdown("---")
    st.markdown(
        f"<h3>{_fa('fa-clock-rotate-left')} &nbsp;Recent Analyses</h3>",
        unsafe_allow_html=True,
    )
    rows = []
    for a in all_analyses[:20]:
        rows.append({
            "Patient ID":  a["patient_id"],
            "Name":        a.get("full_name", "-"),
            "Age":         a.get("age", "-"),
            "Prediction":  a.get("prediction", "-"),
            "BI-RADS":     a.get("birads_category", "-"),
            "Malignant %": f"{a.get('malignant_prob', 0)*100:.1f}%",
            "Date":        a.get("analysed_at", "")[:16],
            "By":          a.get("analysed_by", "-"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# Doctor — All Patients
# ══════════════════════════════════════════════════════════════

def page_all_patients():
    st.markdown(
        f"<h1>{_fa('fa-users', 'fa-red')} &nbsp;All Patient Records</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    search   = st.text_input("Search by name or ID")
    patients = db.search_patients(search) if search else db.get_all_patients()

    if not patients:
        st.info("No patients found.")
        return

    st.markdown(f"**{len(patients)} patient(s)**")

    for p in patients:
        pid = p["patient_id"]
        with st.expander(
            f"{p['full_name']}  |  {pid}  |  Age {p['age']}",
            expanded=False,
        ):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Contact:** {p.get('contact', '-')}")
            c2.markdown(f"**Registered:** {p.get('created_at', '-')[:10]}")
            c3.markdown(f"**By:** {p.get('created_by', '-')}")
            if p.get("history"):
                st.markdown(f"**Clinical History:** {p['history']}")

            analyses = db.get_analyses_for_patient(pid)
            if not analyses:
                st.caption("No analyses for this patient.")
                continue

            for idx, a in enumerate(analyses):
                pred  = a.get("prediction", "-")
                color = "#E50914" if pred == "Malignant" else "#46d369"
                icon  = "fa-triangle-exclamation" if pred == "Malignant" else "fa-circle-check"
                with st.container():
                    st.markdown(
                        f"<div style='border-left:3px solid {color};padding-left:12px;margin:8px 0'>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"{_fa(icon, style=f'color:{color}')} &nbsp;"
                        f"**Analysis #{idx+1}** &nbsp;|&nbsp; `{a['analysed_at'][:16]}`  "
                        f"<span style='color:{color};font-weight:700'>{pred}</span>  "
                        f"| {a.get('birads_category', '-')} "
                        f"| Malignant: {a.get('malignant_prob', 0)*100:.1f}%",
                        unsafe_allow_html=True,
                    )
                    if a.get("features"):
                        tc1, tc2 = st.columns(2)
                        with tc1:
                            st.plotly_chart(viz.radar_chart(a["features"], p["full_name"]),
                                            use_container_width=True, key=f"radar_{pid}_{idx}")
                        with tc2:
                            st.plotly_chart(viz.probability_gauge(
                                a.get("benign_prob", 0), a.get("malignant_prob", 0)),
                                use_container_width=True, key=f"gauge_{pid}_{idx}")
                        st.plotly_chart(viz.feature_bar_chart(a["features"]),
                                        use_container_width=True, key=f"fbar_{pid}_{idx}")

                    img_p = a.get("image_path", "")
                    if img_p and os.path.exists(img_p):
                        st.image(Image.open(img_p).convert("RGB"), caption="Mammogram", width=300)

                    st.markdown("</div>", unsafe_allow_html=True)

                    if st.button(f"Download Report  —  Analysis #{idx+1}",
                                 key=f"rep_{pid}_{idx}"):
                        _generate_and_download_report(p, a)

            st.markdown("---")


# ══════════════════════════════════════════════════════════════
# Doctor — Analysis Detail
# ══════════════════════════════════════════════════════════════

def page_analysis_detail():
    st.markdown(
        f"<h1>{_fa('fa-microscope', 'fa-red')} &nbsp;Analysis Detail</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    all_patients = db.get_all_patients()
    if not all_patients:
        st.info("No patients found.")
        return

    pid_options = [f"{p['patient_id']} - {p['full_name']}" for p in all_patients]
    selected    = st.selectbox("Select Patient", pid_options)
    pid         = selected.split(" - ")[0]
    patient     = db.get_patient(pid)
    analyses    = db.get_analyses_for_patient(pid)

    if not analyses:
        st.info("No analyses for this patient.")
        return

    dates  = [f"#{i+1}: {a['analysed_at'][:16]} - {a['prediction']}"
              for i, a in enumerate(analyses)]
    choice = st.selectbox("Select Analysis", dates)
    idx    = dates.index(choice)
    a      = analyses[idx]

    st.markdown("---")

    pred  = a.get("prediction", "-")
    badge = "badge-malignant" if pred == "Malignant" else "badge-benign"
    icon  = "fa-triangle-exclamation" if pred == "Malignant" else "fa-circle-check"
    st.markdown(
        f"<h2>{_fa(icon, style='color:#E50914' if pred=='Malignant' else 'color:#46d369')} "
        f"&nbsp;Diagnosis: <span class='{badge}'>{pred.upper()}</span></h2>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Benign",    f"{a.get('benign_prob',0)*100:.1f}%")
    c2.metric("Malignant", f"{a.get('malignant_prob',0)*100:.1f}%")
    c3.metric("BI-RADS",   a.get("birads_category", "-"))
    c4.metric("Date",      a.get("analysed_at", "-")[:10])

    st.caption(f"Assessment: {a.get('birads_desc', '-')}")
    if a.get("notes"):
        st.info(f"Notes: {a['notes']}")

    st.markdown("---")

    t1, t2, t3 = st.tabs(["Probability", "Cell Profile", "Features"])

    with t1:
        cc1, cc2 = st.columns(2)
        cc1.plotly_chart(viz.probability_gauge(a.get("benign_prob", 0), a.get("malignant_prob", 0)),
                         use_container_width=True, key="det_gauge")
        cc2.plotly_chart(viz.probability_bar(a.get("benign_prob", 0), a.get("malignant_prob", 0)),
                         use_container_width=True, key="det_pbar")
        st.plotly_chart(viz.birads_distribution_chart(a.get("birads_category", "")),
                        use_container_width=True, key="det_birads")

    with t2:
        if a.get("features"):
            st.plotly_chart(viz.radar_chart(a["features"], patient["full_name"]),
                            use_container_width=True, key="det_radar")
        else:
            st.info("No feature data available.")

    with t3:
        if a.get("features"):
            st.plotly_chart(viz.feature_bar_chart(a["features"]),
                            use_container_width=True, key="det_fbar")
            feats   = a["features"]
            mean_kk = sorted([k for k in feats if k.endswith("_mean")])
            rows    = []
            for k in mean_kk:
                rows.append({
                    "Feature": k.replace("_mean", "").replace("_", " ").title(),
                    "Mean":    round(feats[k], 4),
                    "Worst":   round(feats.get(k.replace("_mean", "_worst"), 0), 4),
                    "SE":      round(feats.get(k.replace("_mean", "_se"), 0), 4),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    img_path = a.get("image_path", "")
    if img_path and os.path.exists(img_path):
        st.markdown("---")
        st.markdown(
            f"<h3>{_fa('fa-x-ray')} &nbsp;Mammogram Image</h3>",
            unsafe_allow_html=True,
        )
        col_img, _ = st.columns([1, 1])
        col_img.image(Image.open(img_path).convert("RGB"),
                      caption="Uploaded Mammogram", width=600)

    st.markdown("---")
    if st.button("Generate & Download PDF Report", type="primary"):
        _generate_and_download_report(patient, a)


# ══════════════════════════════════════════════════════════════
# Report helper
# ══════════════════════════════════════════════════════════════

def _generate_and_download_report(patient: dict, analysis: dict):
    with st.spinner("Generating PDF report..."):
        try:
            figs = {
                "Probability Gauge":    viz.probability_gauge(
                    analysis.get("benign_prob", 0), analysis.get("malignant_prob", 0)),
                "Cell Feature Profile": viz.radar_chart(
                    analysis.get("features", {}), patient.get("full_name", "")),
                "Feature Comparison":   viz.feature_bar_chart(
                    analysis.get("features", {})),
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
            label="Download PDF",
            data=f.read(),
            file_name=Path(path).name,
            mime="application/pdf",
            key=f"dl_{uuid.uuid4().hex[:6]}",
        )
    st.success("Report ready! Click Download PDF above.")


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
        if page == "upload":       page_upload()
        elif page == "my_patients": page_my_patients()
        else:                       page_upload()
    elif role == "doctor":
        if page == "overview":          page_overview()
        elif page == "all_patients":    page_all_patients()
        elif page == "analysis_detail": page_analysis_detail()
        else:                           page_overview()


if __name__ == "__main__":
    main()