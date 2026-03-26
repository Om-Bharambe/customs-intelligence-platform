import os
import sys
import tempfile
from collections import Counter

import streamlit as st
from PIL import Image
from ultralytics import YOLO

sys.path.append("scripts")
from risk_scoring import calculate_risk
from fp_fn_report import get_metrics
from save_feedback import save_feedback

st.set_page_config(
    page_title="Customs Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Helpers ----------
def risk_color(level):
    if str(level).lower() == "high":
        return "#ff4b4b"
    elif str(level).lower() == "medium":
        return "#f39c12"
    return "#2ecc71"

def recommended_action(level):
    if str(level).lower() == "high":
        return "Immediate manual inspection required."
    elif str(level).lower() == "medium":
        return "Send for secondary review."
    return "Clear for normal processing."

def reset_analysis():
    st.session_state.detections = []
    st.session_state.analysis_done = False
    st.session_state.annotated_image = None
    st.session_state.risk_score = "0.0"
    st.session_state.risk_level = "Low"
    st.session_state.explanation = ""
    st.session_state.feedback_submitted = False

# ---------- Session State ----------
if "detections" not in st.session_state:
    st.session_state.detections = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None
if "risk_score" not in st.session_state:
    st.session_state.risk_score = "0.0"
if "risk_level" not in st.session_state:
    st.session_state.risk_level = "Low"
if "explanation" not in st.session_state:
    st.session_state.explanation = ""
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# ---------- Model ----------
MODEL_PATH = "model/best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found in model/best.pt")
    st.stop()

model = YOLO(MODEL_PATH)

# ---------- Sidebar ----------
st.sidebar.title("Control Panel")
st.sidebar.markdown("Adjust the screening settings and view system information.")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.90,
    value=0.15,
    step=0.05
)

user_role = st.sidebar.selectbox(
    "User View",
    ["Officer", "Manager"]
)

st.sidebar.markdown("### System Info")
st.sidebar.write("**Model:** YOLO-based X-ray detector")
st.sidebar.write("**Dataset:** Prototype dataset")
st.sidebar.write("**Target Dataset:** PIDray / CargoXray")

show_metrics = st.sidebar.button("Show Model Evaluation Metrics")

# ---------- Main Header ----------
st.title("Customs Intelligence Platform")
st.caption("AI-assisted screening for suspicious or prohibited items in X-ray scans")
st.divider()

# ---------- Metrics Section ----------
if show_metrics:
    metrics = get_metrics()
    st.subheader("Model Evaluation Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{metrics['Precision']:.3f}")
    m2.metric("Recall", f"{metrics['Recall']:.3f}")
    m3.metric("mAP50", f"{metrics['mAP50']:.3f}")

    m4, m5, m6 = st.columns(3)
    m4.metric("mAP50-95", f"{metrics['mAP50-95']:.3f}")
    m5.metric("Est. False Positives", f"{metrics['False Positives']}")
    m6.metric("Est. False Negatives", f"{metrics['False Negatives']}")
    st.divider()

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_name = uploaded_file.name
    image = Image.open(uploaded_file)

    if st.button("Run Detection", use_container_width=True):
        reset_analysis()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        with st.spinner("Analyzing image..."):
            results = model.predict(source=temp_path, conf=confidence_threshold, device="cpu")
            result = results[0]
            annotated = result.plot()

        st.session_state.annotated_image = annotated
        st.session_state.detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            names = result.names

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = names[class_id]
                st.session_state.detections.append((class_name, confidence))

            score, level = calculate_risk(st.session_state.detections)
            highest_detection = max(st.session_state.detections, key=lambda x: x[1])

            st.session_state.risk_score = score
            st.session_state.risk_level = level
            st.session_state.explanation = (
                f"Image flagged because {highest_detection[0]} was detected "
                f"with confidence {highest_detection[1]:.2f}."
            )
        else:
            st.session_state.risk_score = "0.0"
            st.session_state.risk_level = "Low"
            st.session_state.explanation = "No suspicious object detected."

        st.session_state.analysis_done = True
        os.remove(temp_path)

    # ---------- Alert Banner ----------
    if st.session_state.analysis_done and str(st.session_state.risk_level).lower() == "high":
        st.markdown(
            f"""
            <div style="
                background-color:{risk_color(st.session_state.risk_level)};
                padding:14px;
                border-radius:12px;
                color:white;
                font-weight:600;
                margin-bottom:15px;">
                ⚠ High Risk Shipment Detected — Manual inspection recommended
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------- Image Panels ----------
    left, right = st.columns(2)

    with left:
        st.subheader("📤 Uploaded Scan")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("🧠 Detection Output")
        if st.session_state.analysis_done and st.session_state.annotated_image is not None:
            st.image(st.session_state.annotated_image, use_container_width=True)
        else:
            st.info("Run detection to view the annotated output.")

    # ---------- Summary Cards ----------
    if st.session_state.analysis_done:
        num_items = len(st.session_state.detections)
        highest_conf = max([c for _, c in st.session_state.detections], default=0.0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Level", st.session_state.risk_level)
        c2.metric("Risk Score", st.session_state.risk_score)
        c3.metric("Detected Items", num_items)
        c4.metric("Highest Confidence", f"{highest_conf:.2f}")

        st.divider()

        # ---------- Detected Objects ----------
        st.subheader("📦 Detected Objects")
        if len(st.session_state.detections) > 0:
            counts = Counter([name for name, _ in st.session_state.detections])

            count_cols = st.columns(min(len(counts), 5))
            for i, (item, count) in enumerate(counts.items()):
                count_cols[i % len(count_cols)].metric(f"{item} Count", count)

            rows = []
            for class_name, confidence in st.session_state.detections:
                rows.append({
                    "Object": class_name,
                    "Confidence": round(confidence, 2),
                    "Threat Weight": {
                        "Gun": 10,
                        "Knife": 8,
                        "Scissors": 5,
                        "Pliers": 4,
                        "Wrench": 3
                    }.get(class_name, 1)
                })

            st.dataframe(rows, use_container_width=True)
        else:
            st.success("No suspicious object detected.")

        # ---------- Risk + Explanation + Action ----------
        a, b, c = st.columns(3)

        with a:
            st.subheader("⚠ Risk Assessment")
            st.markdown(
                f"""
                <div style="
                    background-color:{risk_color(st.session_state.risk_level)};
                    padding:16px;
                    border-radius:12px;
                    color:white;
                    text-align:center;">
                    <div style="font-size:18px; font-weight:600;">{st.session_state.risk_level} Risk</div>
                    <div style="font-size:28px; font-weight:700;">{st.session_state.risk_score}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with b:
            st.subheader("📝 Explanation")
            st.info(st.session_state.explanation)

        with c:
            st.subheader("✅ Recommended Action")
            st.warning(recommended_action(st.session_state.risk_level))

        st.divider()

        # ---------- Quick Feedback ----------
        st.subheader("👍 Quick Feedback")
        st.caption("Optional one-click feedback to improve future model performance.")

        feedback_status = st.radio(
            "Was this detection correct?",
            ["Correct", "Incorrect"],
            horizontal=True,
            key="feedback_status"
        )

        issue_type = ""
        note = ""

        low_confidence_case = highest_conf < 0.40
        if low_confidence_case:
            st.info("Low-confidence result detected. Human review is especially useful here.")

        if feedback_status == "Incorrect":
            issue_type = st.selectbox(
                "What was wrong?",
                ["Wrong item", "Missed item", "False alarm", "Wrong bounding box"],
                key="issue_type"
            )
            note = st.text_area(
                "Optional note",
                placeholder="Example: This looks like a knife, not a gun.",
                key="feedback_note"
            )

        if st.button("Submit Feedback"):
            save_feedback(
                image_name=image_name,
                detections=st.session_state.detections,
                feedback_status=feedback_status,
                issue_type=issue_type,
                note=note
            )
            st.session_state.feedback_submitted = True

        if st.session_state.feedback_submitted:
            st.success("Feedback saved successfully.")

        # ---------- History Placeholder ----------
        st.divider()
        st.subheader("🕘 Scan History")
        st.info("History view can be added next to show previous scans, risk levels, and feedback logs.")

        # ---------- Manager View Placeholder ----------
        if user_role == "Manager":
            st.divider()
            st.subheader("📊 Manager View")
            st.info("Manager dashboard can be extended to show aggregated risk trends, frequent detections, and model feedback statistics.")