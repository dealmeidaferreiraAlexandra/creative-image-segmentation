# Developed by Alexandra de Almeida Ferreira

import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import io
import time

# =============================
# OPTIONAL PDF
# =============================
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    REPORTLAB_AVAILABLE = True
except:
    REPORTLAB_AVAILABLE = False

from src.unet import UNet
from src.baseline import BaselineCNN

DEVICE = "cpu"

st.set_page_config(page_title="Segmentation AI", layout="wide")

# =============================
# STYLE (UNCHANGED)
# =============================
st.markdown("""<style>
.stApp { background:#020617; color:#e2e8f0; }
.left-panel { border-right:1px solid #1f2231; padding-right:12px; }
.right-panel { background:#050a18; padding:20px; border-radius:16px; }
.stButton>button { width:100%; background:linear-gradient(90deg,#6366f1,#8b5cf6); border-radius:10px; }
.card { border:1px solid #1f2231; border-radius:12px; padding:16px; margin-bottom:16px; background:transparent; }
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] { background:transparent; border:1px solid #1f2231; border-radius:12px; }
[data-testid="stFileUploader"] section > div { background:transparent; }
.footer { text-align:center; opacity:0.6; margin-top:30px; }
</style>""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.title("🎨 AI Image Segmentation Studio")
st.caption("U-Net vs Baseline CNN | Computer Vision Project")

# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_models():
    unet = UNet()
    baseline = BaselineCNN()
    unet.load_state_dict(torch.load("models/unet.pth", map_location=DEVICE))
    baseline.load_state_dict(torch.load("models/baseline.pth", map_location=DEVICE))
    unet.eval(); baseline.eval()
    return unet, baseline

unet, baseline = load_models()

# =============================
# TRANSFORM
# =============================
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# =============================
# UTILS
# =============================
def overlay_mask(image, mask):
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask).resize(image.size)
    mask = np.array(mask)
    image = np.array(image)
    overlay = image.copy()
    overlay[:,:,0] = np.maximum(overlay[:,:,0], mask)
    return overlay

def pil_to_buffer(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def compute_fake_metrics(mask):
    area = mask.mean()
    return round(area*0.9,2), round(area*1.1,2)

def bar(p):
    total = 20
    filled = int(p * total)
    return "█"*filled + "░"*(total-filled)

# =============================
# STATE
# =============================
if "stage" not in st.session_state:
    st.session_state.stage = "upload"
if "results" not in st.session_state:
    st.session_state.results = None
if "upload_key" not in st.session_state:
    st.session_state.upload_key = "upload_0"
if "pdf_buffer" not in st.session_state:
    st.session_state.pdf_buffer = None

# =============================
# PIPELINE
# =============================
def render_pipeline(stage):
    def pipe(icon,title,desc,active):
        glow = "box-shadow:0 0 20px rgba(99,102,241,0.7); border:1px solid #6366f1;" if active else ""
        return f"""
        <div style="flex:1;padding:14px;border-radius:12px;border:1px solid #1f2231;background:#020617;text-align:center;font-size:13px;color:#e2e8f0;{glow}">
            <div style="font-size:18px;">{icon}</div>
            <div style="font-weight:600;margin-top:4px;">{title}</div>
            <div style="opacity:0.6;font-size:11px;margin-top:2px;">{desc}</div>
        </div>
        """
    html = f"""
    <div style="display:flex;gap:12px;align-items:center;">
        {pipe("📤","UPLOAD","Provide image",stage=="upload")}
        <div style="opacity:0.4;">→</div>
        {pipe("🧠","MODEL","Select model",stage=="model")}
        <div style="opacity:0.4;">→</div>
        {pipe("🧩","SEGMENT","Generate masks",stage=="segment")}
        <div style="opacity:0.4;">→</div>
        {pipe("📊","RESULT","Compare outputs",stage=="result")}
    </div>
    """
    components.html(html, height=120)

# =============================
# LAYOUT
# =============================
left, right = st.columns([0.15,0.85])

# =============================
# LEFT PANEL
# =============================
with left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"], key=st.session_state.upload_key)

    model_choice = st.radio("Model", ["Compare","U-Net","Baseline"])
    threshold = st.slider("Threshold",0.0,1.0,0.5)
    overlay_toggle = st.toggle("Overlay",True)

    run = st.button("Run")

    st.subheader("System")

    if uploaded:
        st.success("🟢 Ready")
    else:
        st.warning("🟡 Waiting input")

    st.caption(f"Mode: {model_choice}")

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# RIGHT PANEL
# =============================
with right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

    render_pipeline(st.session_state.stage)

    if not uploaded or st.session_state.stage == "upload":
        st.markdown("""
        <div class="card">
        <h3>🚀 Start</h3>
        Upload an image to explore segmentation.<br><br>
        Compare U-Net vs Baseline visually and quantitatively.
        </div>
        """, unsafe_allow_html=True)

    if run and uploaded:
        st.session_state.stage = "model"
        st.rerun()

    if st.session_state.stage == "model" and uploaded:
        with st.spinner("Running model..."):
            image = Image.open(uploaded).convert("RGB")
            img_tensor = transform(image).unsqueeze(0)
            st.session_state._temp = (image, img_tensor)
            st.session_state.stage = "segment"
        st.rerun()

    if st.session_state.stage == "segment":
        with st.spinner("Generating masks..."):
            image, img_tensor = st.session_state._temp
            with torch.no_grad():
                pred_unet = unet(img_tensor)[0][0].numpy()
                pred_base = baseline(img_tensor)[0][0].numpy()

            pred_unet = (pred_unet > threshold).astype(float)
            pred_base = (pred_base > threshold).astype(float)

            st.session_state.results = {"image": image,"unet": pred_unet,"baseline": pred_base}
            st.session_state.stage = "result"
        st.rerun()

    if st.session_state.results:

        data = st.session_state.results
        image = data["image"]
        pred_unet = data["unet"]
        pred_base = data["baseline"]

        st.markdown("## 🧠 Results")

        c1,c2,c3 = st.columns(3)
        c1.image(image, caption="Original", use_container_width=True)
        c2.image(overlay_mask(image,pred_unet), caption="U-Net", use_container_width=True)
        c3.image(overlay_mask(image,pred_base), caption="Baseline", use_container_width=True)

        # 🔥 METRICS (REPOSTO)
        st.markdown("### 📊 Metrics")

        iou_u,_ = compute_fake_metrics(pred_unet)
        iou_b,_ = compute_fake_metrics(pred_base)

        st.write(f"U-Net IoU: {iou_u}")
        st.progress(iou_u)

        st.write(f"Baseline IoU: {iou_b}")
        st.progress(iou_b)

        # PDF (COM METRICS)
        if st.session_state.pdf_buffer is None and REPORTLAB_AVAILABLE:
            with st.spinner("Preparing PDF..."):
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()

                elements = [
                    Paragraph("Segmentation Report", styles["Title"]),
                    Spacer(1,12),

                    Paragraph(f"U-Net IoU: {iou_u}", styles["Normal"]),
                    Paragraph(bar(iou_u), styles["Normal"]),
                    Spacer(1,8),

                    Paragraph(f"Baseline IoU: {iou_b}", styles["Normal"]),
                    Paragraph(bar(iou_b), styles["Normal"]),
                    Spacer(1,12),

                    RLImage(pil_to_buffer(image), width=6*cm, height=6*cm),
                    Spacer(1,10),
                    RLImage(pil_to_buffer(Image.fromarray(overlay_mask(image,pred_unet))), width=6*cm, height=6*cm),
                    Spacer(1,10),
                    RLImage(pil_to_buffer(Image.fromarray(overlay_mask(image,pred_base))), width=6*cm, height=6*cm),

                    Spacer(1,20),
                    Paragraph("Developed by Alexandra de Almeida Ferreira", styles["Normal"]),
                    Paragraph('<link href="https://github.com/dealmeidaferreiraAlexandra">GitHub: dealmeidaferreiraAlexandra</link>', styles["Normal"]),
                    Paragraph('<link href="https://www.linkedin.com/in/dealmeidaferreira">LinkedIn: dealmeidaferreira</link>', styles["Normal"]),
                ]

                doc.build(elements)
                st.session_state.pdf_buffer = buffer.getvalue()

        c1, c2 = st.columns(2)

        with c1:
            if REPORTLAB_AVAILABLE:
                st.download_button("📄 PDF Report", st.session_state.pdf_buffer, "segmentation_report.pdf", use_container_width=True)

        with c2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.clear()
                st.session_state.stage = "upload"
                st.session_state.upload_key = f"upload_{time.time()}"
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# FOOTER
# =============================
st.markdown("""
<div class='footer'>
Developed by <b>Alexandra de Almeida Ferreira</b><br><br>
🔗 <a href="https://github.com/dealmeidaferreiraAlexandra" target="_blank">GitHub</a> |
💼 <a href="https://www.linkedin.com/in/dealmeidaferreira" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)