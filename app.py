import os
import math
import tempfile
import torch
import streamlit as st
from PIL import Image

from model_utils import load_checkpoint, predict_video, extract_frames

st.set_page_config(
    page_title="Anime vs Cartoon Classifier",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #05070d;
    color: #f5f5f5;
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #05070d 0%, #0b0f19 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
    color: white;
}

.sub-text {
    color: #b8bfd3;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

.hero-card {
    background: linear-gradient(135deg, #111522 0%, #0c111b 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    margin-bottom: 20px;
}

.metric-card {
    background: #0f1420;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}

.metric-label {
    font-size: 0.9rem;
    color: #9aa3b2;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: white;
}

.result-good {
    color: #7ee787;
}

.result-warn {
    color: #79c0ff;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 14px;
    color: white;
}

.upload-box {
    background: #0d121c;
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 18px;
    margin-top: 8px;
    margin-bottom: 12px;
}

.frame-card {
    background: #0f1420;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 10px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
}

.small-label {
    font-size: 0.85rem;
    color: #9aa3b2;
}

.badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    background: rgba(255,255,255,0.08);
    color: white;
    margin-top: 6px;
}

.final-badge {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 0.95rem;
    font-weight: 700;
    background: linear-gradient(90deg, #1f6feb, #7c3aed);
    color: white;
    margin-top: 6px;
}

hr {
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.07);
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero-card">
    <div class="main-title">🎬 Anime vs Cartoon Video Classifier</div>
    <div class="sub-text">
        Upload a short video clip and the model will analyze sampled frames to classify it as 
        <b>Anime</b> or <b>Cartoon</b>.
   <div style="margin-top: 14px;">
    <div style="color:#9aa3b2; font-size:0.95rem; margin-bottom:10px;">
        Developed by
    </div>
    <div style="display:flex; gap:10px; flex-wrap:wrap;">
        <div class="final-badge">Tanishq Rawat</div>
        <div class="final-badge">Vibhor Malik</div>
        <div class="final-badge">Nicky Cheng</div>
    </div>
</div>
""", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model_once():
    checkpoint_path = "outputs/anime_vs_cartoon_efficientnet_b0.pth"
    return load_checkpoint(checkpoint_path, device)

model, transform, idx_to_class = load_model_once()

# -----------------------------
# Upload Section
# -----------------------------
st.markdown('<div class="section-title">Upload Video</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi", "mkv"],
    help="Best results with short, clear clips between 5–15 seconds."
)

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
        st.video(uploaded_file)

    with col2:
        st.markdown('<div class="section-title">Video Details</div>', unsafe_allow_html=True)
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">File Name</div>
            <div class="metric-value" style="font-size:1.1rem;">{uploaded_file.name}</div>
        </div>
        <br>
        <div class="metric-card">
            <div class="metric-label">File Size</div>
            <div class="metric-value">{file_size_mb:.2f} MB</div>
        </div>
        """, unsafe_allow_html=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_video_path = tmp_file.name

    if st.button("Analyze Video", use_container_width=True):
        with st.spinner("Analyzing frames and generating prediction..."):
            result = predict_video(temp_video_path, model, transform, idx_to_class, device)
            preview_frames = extract_frames(temp_video_path, sample_every=30, max_frames=12)

        st.markdown("<hr>", unsafe_allow_html=True)

        # -----------------------------
        # Prediction Summary
        # -----------------------------
        st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)

        final_label = result["final_label"]
        conf = result["confidence"] * 100
        total_frames = len(result["frame_predictions"])

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Final Prediction</div>
                <div class="metric-value result-good">{final_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value result-warn">{conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Frames Analyzed</div>
                <div class="metric-value">{total_frames}</div>
            </div>
            """, unsafe_allow_html=True)

        st.progress(min(max(result["confidence"], 0.0), 1.0))

        st.markdown(f"""
        <div style="margin-top:12px; margin-bottom:8px;">
            <span class="final-badge">Result: {final_label}</span>
        </div>
        """, unsafe_allow_html=True)

        # -----------------------------
        # Frame Gallery
        # -----------------------------
        st.markdown('<div class="section-title">Sampled Frames</div>', unsafe_allow_html=True)
        st.markdown("These are the frames used to estimate the final video class.")

        frame_preds = result["frame_predictions"]
        num_cols = 4
        rows = math.ceil(len(preview_frames) / num_cols)

        frame_index = 0
        for _ in range(rows):
            cols = st.columns(num_cols)
            for col in cols:
                if frame_index < len(preview_frames):
                    frame_img = preview_frames[frame_index]
                    pred = frame_preds[frame_index] if frame_index < len(frame_preds) else None

                    with col:
                        st.markdown('<div class="frame-card">', unsafe_allow_html=True)
                        st.image(frame_img, use_container_width=True)

                        if pred:
                            st.markdown(
                                f"""
                                <div class="small-label">Frame {frame_index + 1}</div>
                                <div class="badge">{pred['label']} • {pred['confidence'] * 100:.1f}%</div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                frame_index += 1

        # -----------------------------
        # Detailed Frame Results
        # -----------------------------
        with st.expander("Show detailed frame-by-frame results"):
            for i, pred in enumerate(frame_preds, start=1):
                st.write(f"Frame {i}: {pred['label']} ({pred['confidence'] * 100:.2f}%)")

    if os.path.exists(temp_video_path):
        try:
            os.remove(temp_video_path)
        except:
            pass

else:
    st.markdown("""
    <div class="hero-card" style="text-align:center;">
        <div style="font-size:1.2rem; font-weight:700; color:white; margin-bottom:8px;">
            No video uploaded yet
        </div>
        <div style="color:#9aa3b2;">
            Upload a short anime or cartoon video clip to start the analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)