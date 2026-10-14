"""
Streamlit Web Application: Early Crop Disease Detection Using Multimodal Fusion.
Integrates Image Analysis (EfficientNet-B0), Microclimate Sensors (MLP & RF),
Feature-Level Fusion, Grad-CAM Explainability, and Agronomic Disease-Risk Advisory.
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import torch

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference.predict import DiseasePredictor
from src.explainability.gradcam import GradCAM
from src.data.dataset import CLASS_NAMES, FEATURE_NAMES
from src.utils.history_db import log_prediction, fetch_prediction_history
from src.iot.sensor_receiver import get_latest_telemetry, store_telemetry_reading

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Crop Disease Multimodal Fusion",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Premium Custom Styling & CSS Design System
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(6, 78, 59, 0.3);
    }
    .main-header h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 8px 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin: 0;
        max-width: 850px;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        backdrop-filter: blur(10px);
    }

    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-healthy { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
    .badge-anthracnose { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }
    .badge-leafspot { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }

    .stat-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10b981;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-label {
        font-size: 0.82rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Predictor Instance
# -----------------------------------------------------------------------------
@st.cache_resource
def get_predictor():
    return DiseasePredictor()


predictor = get_predictor()

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?w=400&auto=format&fit=crop&q=60", use_container_width=True)
    st.markdown("### 🌿 Multimodal AI System")
    st.markdown("**Crop:** *Syzygium cumini* (Jamun)")
    st.markdown("**Dataset:** DA-IICT Gandhinagar (Benchmark)")
    
    selected_page = st.radio(
        "Navigation",
        [
            "Home Dashboard",
            "Disease Detection",
            "Prediction Explanation (Grad-CAM)",
            "Environmental Monitoring",
            "Model Evaluation & Ablation",
            "Prediction History",
            "IoT & ESP32 Hub"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### Hardware Accelerator")
    device_name = str(predictor.device).upper()
    if "MPS" in device_name:
        st.success(f"⚡ Apple Silicon MPS Accelerated ({device_name})")
    elif "CUDA" in device_name:
        st.success(f"🚀 NVIDIA CUDA Accelerated ({device_name})")
    else:
        st.info(f"💻 CPU Execution ({device_name})")

    st.caption("v1.0.0 • Research Project Edition")


# =============================================================================
# PAGE 1: HOME DASHBOARD
# =============================================================================
if selected_page == "Home Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>Early Crop Disease Detection via Multimodal Fusion</h1>
        <p>A research-grade AI framework combining high-resolution leaf RGB imagery with in-situ microclimate environmental sensor telemetry for robust plant pathology classification and early risk advisory.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-val">3</div>
            <div class="stat-label">Target Classes</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-val">5</div>
            <div class="stat-label">Microclimate Sensors</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-val">4,195</div>
            <div class="stat-label">DA-IICT Dataset Size</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-val">EfficientNet-B0</div>
            <div class="stat-label">Visual Backbone</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🔬 Central Research Question")
    st.info(
        "**“Does multimodal fusion of leaf images and environmental measurements improve disease classification "
        "and early-warning performance compared with standalone image-only and sensor-only baselines?”**\n\n"
        "*Scientific Distinction:* Environmental conditions alone represent microclimate pathogen favorability, "
        "not diagnostic proof of physical infection. This system integrates visual symptoms with microclimate context."
    )

    st.markdown("### 🏛️ Multimodal Fusion Architecture")
    st.markdown("""
    ```mermaid
    graph LR
        A[RGB Leaf Image 224x224] --> B[EfficientNet-B0 Backbone]
        B --> C[Visual Embedding: 128-d]
        D[5 Environmental Sensors] --> E[Sensor MLP: 64-d]
        E --> F[Sensor Embedding: 32-d]
        C --> G[Feature Concatenation: 160-d]
        F --> G
        G --> H[Fusion Classification Head]
        H --> I[Logits: Healthy / Anthracnose / Leaf Spot]
    ```
    """)

    st.markdown("### 📊 Dataset Integrity & Class Distribution")
    dist_col1, dist_col2 = st.columns([1, 1])

    with dist_col1:
        st.markdown("#### DA-IICT Gandhinagar Reference Distribution")
        df_dist = pd.DataFrame({
            "Disease Class": ["Anthracnose", "Healthy", "Leaf Spot"],
            "Samples": [2389, 1148, 658],
            "Proportion": [56.9, 27.4, 15.7]
        })
        fig = px.pie(
            df_dist, values="Samples", names="Disease Class",
            color="Disease Class",
            color_discrete_map={"Healthy": "#10b981", "Anthracnose": "#ef4444", "Leaf Spot": "#f59e0b"},
            hole=0.45,
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with dist_col2:
        st.markdown("#### Active Local Workspace Data Status")
        if os.path.exists("data/metadata.csv"):
            df_local = pd.read_csv("data/metadata.csv")
            counts = df_local["label"].value_counts().reset_index()
            counts.columns = ["Disease Class", "Samples"]
            is_synthetic = "dataset_origin" in df_local.columns and "SYNTHETIC" in str(df_local["dataset_origin"].iloc[0])
            if is_synthetic:
                st.warning("⚠️ Currently using synthetic verification dataset for local pipeline testing.")
            else:
                st.success("✅ Real DA-IICT dataset mounted locally.")
            fig_local = px.bar(
                counts, x="Disease Class", y="Samples", color="Disease Class",
                color_discrete_map={"Healthy": "#10b981", "Anthracnose": "#ef4444", "Leaf Spot": "#f59e0b"},
                text="Samples"
            )
            fig_local.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
            st.plotly_chart(fig_local, use_container_width=True)
        else:
            st.warning("No local dataset detected. Generate demo data or mount real dataset.")


# =============================================================================
# PAGE 2: DISEASE DETECTION
# =============================================================================
elif selected_page == "Disease Detection":
    st.markdown("## 🔍 Multimodal Disease Diagnosis")
    st.write("Diagnose Jamun foliage by providing a leaf photograph along with environmental sensor measurements.")

    col_input_img, col_input_sensor = st.columns([1, 1])

    with col_input_img:
        st.markdown("### 1. Leaf Image Input")
        uploaded_image = st.file_uploader("Upload leaf photograph (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])
        
        # Quick sample selector if demo data exists
        demo_images = []
        if os.path.exists("data/metadata.csv"):
            meta = pd.read_csv("data/metadata.csv")
            demo_images = meta["image_path"].head(6).tolist()

        selected_demo_img = None
        if demo_images:
            use_demo = st.checkbox("Or select a sample from local dataset", value=(uploaded_image is None))
            if use_demo:
                selected_demo_img = st.selectbox("Sample leaf:", demo_images)

        active_img = None
        if uploaded_image is not None:
            active_img = Image.open(uploaded_image).convert("RGB")
            st.image(active_img, caption="Uploaded Leaf Image", use_container_width=True)
        elif selected_demo_img:
            abs_demo_path = os.path.join("data", selected_demo_img)
            if os.path.exists(abs_demo_path):
                active_img = Image.open(abs_demo_path).convert("RGB")
                st.image(active_img, caption=f"Dataset Sample: {selected_demo_img}", use_container_width=True)

    with col_input_sensor:
        st.markdown("### 2. Environmental Measurements")
        input_method = st.radio("Sensor Input Mode:", ["Manual Entry", "Load from Active CSV Row", "Omit Sensors (Image Only)"], horizontal=True)

        sensor_dict = None
        if input_method == "Manual Entry":
            st.markdown("Provide canopy microclimate sensor measurements:")
            c1, c2 = st.columns(2)
            with c1:
                t_amb = st.slider("Ambient Temperature (°C)", 10.0, 48.0, 28.5, 0.5)
                rh = st.slider("Relative Humidity (%)", 15.0, 98.0, 85.0, 1.0)
                lw = st.slider("Leaf Wetness (hours)", 0.0, 24.0, 8.0, 0.5)
            with c2:
                t_soil = st.slider("Soil Temperature (°C)", 10.0, 42.0, 26.0, 0.5)
                sm = st.slider("Soil Moisture (%)", 5.0, 95.0, 65.0, 1.0)

            sensor_dict = {
                "ambient_temperature": float(t_amb),
                "soil_temperature": float(t_soil),
                "humidity": float(rh),
                "soil_moisture": float(sm),
                "leaf_wetness": float(lw),
            }

        elif input_method == "Load from Active CSV Row":
            if os.path.exists("data/metadata.csv"):
                df_meta = pd.read_csv("data/metadata.csv")
                row_idx = st.number_input("Select row index:", min_value=0, max_value=len(df_meta)-1, value=0)
                selected_row = df_meta.iloc[row_idx]
                sensor_dict = {feat: float(selected_row[feat]) for feat in FEATURE_NAMES}
                st.json(sensor_dict)
            else:
                st.error("metadata.csv not found.")
        else:
            sensor_dict = None
            st.info("Sensor readings omitted. Inference will execute using Image-Only baseline.")

    st.markdown("---")
    st.markdown("### 3. Model Architecture Selection")
    col_model, col_run = st.columns([2, 1])

    with col_model:
        model_mode = st.selectbox(
            "Select Diagnostic Pipeline:",
            [
                ("multimodal", "Feature-Level Multimodal Fusion (EfficientNet-B0 + Sensor MLP)"),
                ("image_only", "Image-Only Baseline (EfficientNet-B0)"),
                ("late_fusion", "Late Fusion Ensemble (Probability Calibration)"),
                ("sensor_only_mlp", "Sensor-Only Baseline (PyTorch MLP)"),
                ("sensor_only_rf", "Sensor-Only Baseline (Random Forest)"),
            ],
            format_func=lambda x: x[1]
        )[0]

    with col_run:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Run Diagnosis", type="primary", use_container_width=True)

    if run_btn:
        if active_img is None and "sensor_only" not in model_mode:
            st.error("Please upload or select a leaf image.")
        else:
            with st.spinner("Executing model inference..."):
                try:
                    result = predictor.predict(
                        image=active_img,
                        sensor_dict=sensor_dict,
                        mode=model_mode,
                    )
                    log_prediction(result, sensor_dict)

                    st.markdown("---")
                    st.markdown("### 📋 Diagnostic Results")

                    if result.get("fallback_occurred"):
                        st.warning(f"⚠️ {result['warning']}")

                    res_col1, res_col2 = st.columns([1, 1])

                    with res_col1:
                        pred_class = result["predicted_class"]
                        conf = result["confidence"] * 100.0
                        badge_class = f"badge-{pred_class.lower().replace(' ', '')}"

                        st.markdown(f"""
                        <div class="glass-card" style="border-left: 6px solid #10b981;">
                            <h3 style="margin-top:0;">Diagnosis: <span class="metric-badge {badge_class}">{pred_class}</span></h3>
                            <p style="font-size: 1.1rem; margin: 8px 0;">Model: <strong>{result['model_used']}</strong></p>
                            <p style="font-size: 1.1rem; margin: 8px 0;">Confidence: <strong>{conf:.1f}%</strong></p>
                            <p style="color: #9ca3af; font-size: 0.85rem; margin: 0;">Inference Latency: {result['latency_ms']} ms</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Class probability distribution chart
                        prob_df = pd.DataFrame(list(result["probabilities"].items()), columns=["Class", "Probability"])
                        prob_df["Probability"] = prob_df["Probability"] * 100.0
                        fig_prob = px.bar(
                            prob_df, x="Class", y="Probability", color="Class",
                            color_discrete_map={"Healthy": "#10b981", "Anthracnose": "#ef4444", "Leaf Spot": "#f59e0b"},
                            text="Probability", range_y=[0, 105],
                        )
                        fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig_prob.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260)
                        st.plotly_chart(fig_prob, use_container_width=True)

                    with res_col2:
                        st.markdown("#### Environmental Disease-Risk Context")
                        risk_info = result.get("risk_assessment")
                        if risk_info:
                            risk_score = risk_info["composite_risk_score"]
                            risk_cat = risk_info["risk_category"]
                            cat_color = risk_info["category_color"]

                            st.markdown(f"""
                            <div class="glass-card" style="border-left: 6px solid {cat_color};">
                                <h4 style="margin-top:0; color: {cat_color};">Microclimate Risk: {risk_cat} ({risk_score}%)</h4>
                                <p style="font-size: 0.95rem;">{risk_info['advisory']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                            if risk_info["factor_alerts"]:
                                st.markdown("**Conducive Environmental Factors:**")
                                for alert in risk_info["factor_alerts"]:
                                    st.markdown(f"- ⚠️ {alert}")

                            st.caption(f"ℹ️ {risk_info['disclaimer']}")
                        else:
                            st.info("No sensor measurements provided. Risk evaluation is inactive.")

                except Exception as e:
                    st.error(f"Inference failed: {str(e)}")


# =============================================================================
# PAGE 3: PREDICTION EXPLANATION (GRAD-CAM)
# =============================================================================
elif selected_page == "Prediction Explanation (Grad-CAM)":
    st.markdown("## 🔬 Visual Explainability via Grad-CAM")
    st.write(
        "Analyze the visual regions within the leaf photograph that contributed to the model's prediction "
        "using Gradient-weighted Class Activation Mapping (Grad-CAM) hooked to EfficientNet-B0."
    )

    col_cam_img, col_cam_ctrl = st.columns([1, 1])

    with col_cam_img:
        sample_img = None
        if os.path.exists("data/metadata.csv"):
            df_meta = pd.read_csv("data/metadata.csv")
            diseased = df_meta[df_meta["label"] != "Healthy"]["image_path"].head(10).tolist()
            if diseased:
                sel_path = st.selectbox("Select sample leaf for explainability:", diseased)
                full_p = os.path.join("data", sel_path)
                if os.path.exists(full_p):
                    sample_img = Image.open(full_p).convert("RGB")
                    st.image(sample_img, caption="Original Input Leaf", use_container_width=True)

    with col_cam_ctrl:
        st.markdown("### Grad-CAM Parameters")
        alpha = st.slider("Heatmap Overlay Transparency (α):", 0.1, 0.9, 0.45, 0.05)
        colormap_choice = st.selectbox("Colormap:", ["jet", "viridis", "plasma", "magma", "turbo"], index=0)
        explain_btn = st.button("Generate Activation Map", type="primary")

    if explain_btn and sample_img is not None:
        with st.spinner("Computing convolutional activations and gradients..."):
            try:
                img_model = predictor.load_image_model()
                gradcam = GradCAM(img_model)

                t_tensor = predictor.image_transforms(sample_img).unsqueeze(0).to(predictor.device)
                heatmap = gradcam.generate_heatmap(t_tensor)
                blended = gradcam.overlay_on_image(sample_img, heatmap, alpha=alpha, colormap=colormap_choice)

                res_c1, res_c2 = st.columns(2)
                with res_c1:
                    st.image(sample_img, caption="Original Leaf", use_container_width=True)
                with res_c2:
                    st.image(blended, caption=f"Grad-CAM Heatmap ({colormap_choice})", use_container_width=True)

                st.markdown("""
                <div class="glass-card">
                    <h4>Pathological Interpretation</h4>
                    <p>The high-intensity red/yellow regions denote pixels that generated the highest positive gradient 
                    activations in the final convolutional layer of EfficientNet-B0. These correspond to necrotic lesions, 
                    chlorotic margins, or circular fungal spots on the leaf blade.</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Grad-CAM generation failed: {str(e)}")


# =============================================================================
# PAGE 4: ENVIRONMENTAL MONITORING
# =============================================================================
elif selected_page == "Environmental Monitoring":
    st.markdown("## 🌡️ Canopy Microclimate Environmental Monitoring")
    st.write("Live sensor telemetry gauges, historical trends, and transparent agronomic disease-risk estimation.")

    telemetry = get_latest_telemetry()
    t_data = telemetry.get("telemetry", {}) if telemetry else {}

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Ambient Temp", f"{t_data.get('ambient_temperature', 28.5):.1f} °C", "Normal")
    with c2:
        st.metric("Soil Temp", f"{t_data.get('soil_temperature', 26.0):.1f} °C", "Normal")
    with c3:
        st.metric("Humidity", f"{t_data.get('humidity', 84.0):.1f} %", "High")
    with c4:
        st.metric("Soil Moisture", f"{t_data.get('soil_moisture', 62.0):.1f} %", "Optimal")
    with c5:
        st.metric("Leaf Wetness", f"{t_data.get('leaf_wetness', 7.5):.1f} h", "High Risk")

    st.markdown("---")
    st.markdown("### 📈 Microclimate Trends Across Recorded Samples")

    if os.path.exists("data/metadata.csv"):
        df_meta = pd.read_csv("data/metadata.csv")
        fig_trend = px.scatter(
            df_meta, x="ambient_temperature", y="humidity",
            color="label", size="leaf_wetness",
            hover_data=["soil_moisture", "plant_id"],
            color_discrete_map={"Healthy": "#10b981", "Anthracnose": "#ef4444", "Leaf Spot": "#f59e0b"},
            title="Environmental Space: Temperature vs Humidity vs Leaf Wetness"
        )
        fig_trend.update_layout(height=450)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No metadata available to plot trends.")


# =============================================================================
# PAGE 5: MODEL EVALUATION & ABLATION
# =============================================================================
elif selected_page == "Model Evaluation & Ablation":
    st.markdown("## 📊 Research Experiments & Model Evaluation")
    st.write("Performance metrics on the held-out test population across all baselines and multimodal architectures.")

    eval_summary_path = "outputs/metrics/evaluation_summary.json"
    ablation_path = "outputs/metrics/ablation_study.json"

    if os.path.exists(eval_summary_path):
        with open(eval_summary_path, "r") as f:
            eval_data = json.load(f)

        table_rows = []
        for m_name, m_stats in eval_data.items():
            table_rows.append({
                "Model Pipeline": m_name,
                "Accuracy": f"{m_stats['accuracy']*100:.2f}%",
                "Macro Precision": f"{m_stats['macro_precision']*100:.2f}%",
                "Macro Recall": f"{m_stats['macro_recall']*100:.2f}%",
                "Macro F1-Score": f"{m_stats['macro_f1']*100:.2f}%",
                "ROC-AUC (OvR)": f"{m_stats['roc_auc_ovr']:.4f}" if m_stats.get('roc_auc_ovr') is not None else "N/A",
                "Latency (ms)": f"{m_stats.get('latency_ms_per_sample', 0):.2f}",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        if os.path.exists(ablation_path):
            with open(ablation_path, "r") as f:
                abl_data = json.load(f)
            st.markdown("### 🔬 Ablation Analysis: Multimodal vs Image-Only")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-val">{abl_data.get('delta_macro_f1', 0.0):+.4f}</div>
                    <div class="stat-label">Δ Macro F1 (Multimodal - Image-Only)</div>
                </div>
                """, unsafe_allow_html=True)
            with col_a2:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-val">{abl_data.get('delta_accuracy', 0.0):+.4f}</div>
                    <div class="stat-label">Δ Accuracy (Multimodal - Image-Only)</div>
                </div>
                """, unsafe_allow_html=True)

            st.info(f"**Scientific Interpretation:** {abl_data.get('scientific_interpretation')}")

        st.markdown("### 🖼️ Evaluation Artifacts & Plots")
        tab1, tab2, tab3, tab4 = st.tabs(["Metrics Comparison", "Confusion Matrices", "ROC Curves", "Loss Progression"])

        with tab1:
            if os.path.exists("outputs/plots/metrics_comparison.png"):
                st.image("outputs/plots/metrics_comparison.png", use_container_width=True)
        with tab2:
            if os.path.exists("outputs/plots/confusion_matrices.png"):
                st.image("outputs/plots/confusion_matrices.png", use_container_width=True)
        with tab3:
            if os.path.exists("outputs/plots/roc_curves.png"):
                st.image("outputs/plots/roc_curves.png", use_container_width=True)
        with tab4:
            if os.path.exists("outputs/plots/loss_curves.png"):
                st.image("outputs/plots/loss_curves.png", use_container_width=True)
    else:
        st.warning("No evaluation metrics found. Run python src/training/evaluate.py to generate test results.")


# =============================================================================
# PAGE 6: PREDICTION HISTORY
# =============================================================================
elif selected_page == "Prediction History":
    st.markdown("## 📜 Prediction History")
    st.write("Local database log of user diagnoses with filter and CSV export options.")

    hist_df = fetch_prediction_history()
    if not hist_df.empty:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_cls = st.multiselect("Filter by Predicted Class:", CLASS_NAMES, default=CLASS_NAMES)
        with col_f2:
            csv_data = hist_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export History to CSV", csv_data, "prediction_history.csv", "text/csv")

        filtered = hist_df[hist_df["predicted_class"].isin(filter_cls)]
        st.dataframe(filtered, use_container_width=True)
    else:
        st.info("No prediction records logged yet. Run a diagnosis in the Disease Detection tab.")


# =============================================================================
# PAGE 7: IOT & ESP32 HUB
# =============================================================================
elif selected_page == "IoT & ESP32 Hub":
    st.markdown("## 📡 ESP32 Hardware Integration Hub")
    st.write("Interface and electrical documentation for the ESP32 microclimate environmental node.")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("""
        ### Hardware Specifications
        - **Microcontroller:** ESP32-WROOM-32 (Dual-core 240MHz, Wi-Fi & BLE)
        - **Sensors:**
          - `DHT22` -> Ambient Air Temp & Relative Humidity (GPIO 4)
          - `DS18B20` -> Soil Root-Zone Temperature (GPIO 15 - OneWire)
          - `Capacitive v1.2` -> Soil Moisture Content (GPIO 34 - ADC1)
          - `Leaf Wetness Grid` -> Surface Water Film Duration (GPIO 35 - ADC1)
        - **Protocol:** HTTP POST JSON payload to `/api/telemetry`
        - **Firmware Location:** `esp32/sensor_firmware.ino`
        """)

    with col_h2:
        st.markdown("### Simulated Telemetry Injection")
        st.write("Test receiver ingestion directly without physical hardware:")
        
        test_t = st.number_input("Test Ambient Temp (°C)", 10.0, 45.0, 29.0)
        test_rh = st.number_input("Test Humidity (%)", 20.0, 98.0, 88.0)
        test_lw = st.number_input("Test Leaf Wetness (h)", 0.0, 24.0, 9.5)

        if st.button("Transmit Test Telemetry Packet"):
            packet = {
                "device_id": "TEST_VIRTUAL_ESP32",
                "hardware_mode": "SIMULATED_TEST",
                "telemetry": {
                    "ambient_temperature": test_t,
                    "soil_temperature": test_t - 2.5,
                    "humidity": test_rh,
                    "soil_moisture": 70.0,
                    "leaf_wetness": test_lw,
                }
            }
            res = store_telemetry_reading(packet)
            st.success(f"Packet received and stored in SQLite database! (Device: {res['device_id']})")
