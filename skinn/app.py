import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ---------------------------------------------------------------- config
DATASET_URL = "https://drive.google.com/drive/folders/1lAgzbmSgU8sM3KqyoidOlLDa73Zc7adj?usp=sharing"
MODEL_PATH = Path(__file__).parent / "skin_cancer_model.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    'Acne and Rosacea Photos',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Atopic Dermatitis Photos',
    'Cellulitis Impetigo and other Bacterial Infections',
    'Eczema Photos',
    'Exanthems and Drug Eruptions',
    'Herpes HPV and other STDs Photos',
    'Light Diseases and Disorders of Pigmentation',
    'Lupus and other Connective Tissue diseases',
    'Melanoma Skin Cancer Nevi and Moles',
    'Poison Ivy Photos and other Contact Dermatitis',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Seborrheic Keratoses and other Benign Tumors',
    'Systemic Disease',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Urticaria Hives',
    'Vascular Tumors',
    'Vasculitis Photos',
    'Warts Molluscum and other Viral Infections',
]
# Classes that deserve a stronger "see a doctor" message
HIGH_RISK = {CLASS_NAMES[1], CLASS_NAMES[9]}

st.set_page_config(page_title="DermaScan AI", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1150px;}
.hero {background: linear-gradient(135deg,#0f766e 0%,#2563eb 100%);
       padding: 2rem 2.2rem; border-radius: 20px; color: #fff; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2.2rem; color:#fff;}
.hero p {margin: .4rem 0 0; opacity: .92; font-size: 1.05rem;}
.card {background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.25);
       border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;}
.pred-label {font-size: 1.35rem; font-weight: 700; line-height: 1.3;}
.pill {display:inline-block; padding: .2rem .8rem; border-radius: 999px;
       font-size: .85rem; font-weight: 600; color:#fff;}
.pill.hi {background:#16a34a;} .pill.mid {background:#d97706;} .pill.lo {background:#dc2626;}
.warn {border-left: 5px solid #dc2626; background: rgba(220,38,38,.10);
       padding: .8rem 1rem; border-radius: 8px; margin: .6rem 0;}
.footer {text-align:center; opacity:.65; font-size:.85rem; margin-top:2rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    # The model has its own Rescaling(1/255) layer, so feed raw 0-255 pixels.
    return np.expand_dims(np.asarray(img, dtype="float32"), 0)


def predict(model, img: Image.Image) -> np.ndarray:
    # The model's last layer is already softmax -> do NOT apply softmax again.
    probs = model.predict(preprocess(img), verbose=0)[0]
    return probs / probs.sum()


def short(name: str, n: int = 48) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📂 Dataset")
    st.write("Download the image dataset used for this project, then upload "
             "any image from it to test the model.")
    st.link_button("⬇️ Open Dataset (Google Drive)", DATASET_URL,
                   use_container_width=True, type="primary")
    st.divider()
    st.header("ℹ️ About")
    st.write(f"**Classes:** {len(CLASS_NAMES)}  \n**Input size:** 224×224 RGB  \n"
             "**Model:** CNN (3 conv blocks + dense head)")
    top_k = st.slider("Predictions to show", 3, 8, 5)
    with st.expander("All classes"):
        for c in CLASS_NAMES:
            st.markdown(f"- {c}")

# ---------------------------------------------------------------- header
st.markdown("""
<div class="hero">
  <h1>🩺 DermaScan AI</h1>
  <p>Upload a skin image and the model will estimate which of 19 skin-condition
  categories it most resembles.</p>
</div>
""", unsafe_allow_html=True)

try:
    model = load_model()
except Exception as e:
    st.error("Could not load the model.")
    st.code(f"{type(e).__name__}: {e}")
    st.info(f"Make sure `{MODEL_PATH.name}` is in the same folder as `app.py` and that "
            "`tensorflow>=2.20` and `keras>=3.13` are installed "
            "(`pip install -r requirements.txt`).")
    st.stop()

tab_single, tab_batch = st.tabs(["🔍 Single image", "🗂️ Batch from dataset"])

# ---------------------------------------------------------------- single
with tab_single:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        up = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png", "webp"])
        if up:
            image = Image.open(up)
            st.image(image, caption="Uploaded image", use_container_width=True)
        else:
            st.markdown('<div class="card">👆 Upload a JPG/PNG image to begin. '
                        'Need images? Use the <b>Dataset</b> button in the sidebar.</div>',
                        unsafe_allow_html=True)

    with right:
        if up:
            with st.spinner("Analysing image…"):
                probs = predict(model, image)
            order = np.argsort(probs)[::-1]
            best = int(order[0]); conf = float(probs[best]) * 100
            level, text = (("hi", "High confidence") if conf >= 70 else
                           ("mid", "Moderate confidence") if conf >= 40 else
                           ("lo", "Low confidence"))
            st.markdown(f"""
            <div class="card">
              <div style="opacity:.7">Most likely category</div>
              <div class="pred-label">{CLASS_NAMES[best]}</div><br>
              <span class="pill {level}">{text} · {conf:.1f}%</span>
            </div>""", unsafe_allow_html=True)
            if CLASS_NAMES[best] in HIGH_RISK:
                st.markdown('<div class="warn">⚠️ This category includes potentially '
                            'malignant lesions. Please consult a dermatologist.</div>',
                            unsafe_allow_html=True)
            if conf < 40:
                st.warning("The model is unsure. Try a clearer, closer, well-lit photo.")

            st.subheader(f"Top {top_k} predictions")
            df = pd.DataFrame({
                "Condition": [short(CLASS_NAMES[i]) for i in order[:top_k]],
                "Probability (%)": [round(float(probs[i]) * 100, 2) for i in order[:top_k]],
            })
            st.bar_chart(df, x="Condition", y="Probability (%)", horizontal=True)
            st.dataframe(df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- batch
with tab_batch:
    st.write("Select several images (for example from the downloaded dataset) "
             "to classify them all at once.")
    files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png", "webp"],
                             accept_multiple_files=True, key="batch")
    if files:
        rows, bar = [], st.progress(0.0)
        for i, f in enumerate(files):
            try:
                p = predict(model, Image.open(f))
                k = int(np.argmax(p))
                rows.append({"File": f.name, "Prediction": CLASS_NAMES[k],
                             "Confidence (%)": round(float(p[k]) * 100, 2)})
            except Exception as e:
                rows.append({"File": f.name, "Prediction": f"Error: {e}", "Confidence (%)": None})
            bar.progress((i + 1) / len(files))
        res = pd.DataFrame(rows)
        st.dataframe(res, hide_index=True, use_container_width=True)
        st.download_button("Download results (CSV)", res.to_csv(index=False).encode(),
                           "predictions.csv", "text/csv")

st.markdown('<div class="footer">⚠️ Educational project only — not a medical device. '
            'Results are not a diagnosis; always consult a qualified doctor.</div>',
            unsafe_allow_html=True)
