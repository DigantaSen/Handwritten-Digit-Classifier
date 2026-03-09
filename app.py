import streamlit as st
import numpy as np
from PIL import Image
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (heavy libs) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    """Load all three saved Keras models once and cache them."""
    from tensorflow.keras.models import load_model  # noqa: E402

    base = os.path.join(os.path.dirname(__file__), "models")
    models = {}
    for name, fname in [
        ("Perceptron", "perceptron_model.h5"),
        ("ANN",        "ann_model.h5"),
        ("CNN",        "cnn_model.h5"),
    ]:
        path = os.path.join(base, fname)
        if os.path.exists(path):
            models[name] = load_model(path)
        else:
            models[name] = None
    return models


# ── Helper ────────────────────────────────────────────────────────────────────
def preprocess_canvas(img_array: np.ndarray) -> np.ndarray:
    """
    Convert a raw RGBA canvas array (H×W×4) to a normalised 28×28 float array.
    Returns shape (28,28).
    """
    # RGBA → grayscale
    img = Image.fromarray(img_array.astype(np.uint8)).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    return arr


def preprocess_upload(uploaded_file) -> np.ndarray:
    """
    Convert an uploaded image file to a normalised 28×28 float array.
    """
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    return arr


def predict_all(models: dict, arr_28x28: np.ndarray) -> dict:
    """
    Run all three models and return {model_name: (digit, confidence_array)}.
    """
    results = {}
    flat   = arr_28x28.reshape(1, 28, 28)         # for Perceptron / ANN
    volume = arr_28x28.reshape(1, 28, 28, 1)       # for CNN

    for name, model in models.items():
        if model is None:
            results[name] = None
            continue

        inp   = volume if name == "CNN" else flat
        probs = model.predict(inp, verbose=0)[0]   # shape (10,)
        digit = int(np.argmax(probs))
        results[name] = (digit, probs)

    return results


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔢 MNIST Classifier")
    st.markdown(
        """
        Draw a digit **0–9** on the canvas (or upload an image) and click
        **Predict** to see how three neural-network architectures compare.

        | Model | Architecture |
        |-------|-------------|
        | **Perceptron** | Flatten → Dense(10, softmax) |
        | **ANN** | Flatten → Dense(256) → Dense(128) → Dense(64) → Dense(32) → Dense(10) |
        | **CNN** | Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → Flatten → Dense(128) → Dropout(0.5) → Dense(10) |
        """
    )
    st.divider()
    input_mode = st.radio(
        "Input mode",
        ["✏️ Draw on canvas", "📁 Upload image"],
        index=0,
    )
    st.divider()
    st.caption("Run the notebook first to generate the `models/` folder.")


# ── Load models ───────────────────────────────────────────────────────────────
models = load_models()
missing = [n for n, m in models.items() if m is None]
if missing:
    st.error(
        f"⚠️  Missing saved model(s): **{', '.join(missing)}**.\n\n"
        "Please run all cells in **project.ipynb** first to train and save the models."
    )
    st.stop()


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("MNIST Handwritten Digit Classifier")
st.markdown("Compare **Perceptron**, **ANN**, and **CNN** predictions side-by-side.")
st.divider()

arr_28x28 = None

# ── Canvas input ──────────────────────────────────────────────────────────────
if input_mode == "✏️ Draw on canvas":
    try:
        from streamlit_drawable_canvas import st_canvas

        col_canvas, col_gap, col_preview = st.columns([3, 0.5, 1.5])

        with col_canvas:
            st.subheader("Draw here")
            canvas_result = st_canvas(
                fill_color   ="rgba(0,0,0,1)",
                stroke_width =20,
                stroke_color ="white",
                background_color="black",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )

        if canvas_result.image_data is not None:
            img_arr = canvas_result.image_data          # (280, 280, 4) RGBA
            arr_28x28 = preprocess_canvas(img_arr)

            with col_preview:
                st.subheader("Preview (28×28)")
                preview = Image.fromarray(
                    (arr_28x28 * 255).astype(np.uint8), mode="L"
                )
                st.image(preview, width=140, caption="Model input")

    except ImportError:
        st.warning(
            "**streamlit-drawable-canvas** is not installed.\n\n"
            "Run:  `pip install streamlit-drawable-canvas`  then restart the app.\n\n"
            "Alternatively switch to **Upload image** mode."
        )

# ── Upload input ──────────────────────────────────────────────────────────────
else:
    col_up, col_gap, col_preview = st.columns([3, 0.5, 1.5])

    with col_up:
        st.subheader("Upload a digit image")
        uploaded = st.file_uploader(
            "PNG / JPG / BMP (ideally white digit on black background)",
            type=["png", "jpg", "jpeg", "bmp"],
        )

    if uploaded:
        arr_28x28 = preprocess_upload(uploaded)

        with col_preview:
            st.subheader("Preview (28×28)")
            preview = Image.fromarray(
                (arr_28x28 * 255).astype(np.uint8), mode="L"
            )
            st.image(preview, width=140, caption="Model input")


# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

if predict_btn and arr_28x28 is not None:
    # Check if canvas is blank (sum is basically 0)
    if arr_28x28.sum() < 0.01:
        st.warning("Canvas appears empty — please draw a digit first.")
        st.stop()

    results = predict_all(models, arr_28x28)

    st.subheader("Predictions")
    cols = st.columns(3)

    model_colors = {
        "Perceptron": "#FF6B6B",
        "ANN":        "#4ECDC4",
        "CNN":        "#45B7D1",
    }

    for col, (name, result) in zip(cols, results.items()):
        digit, probs = result
        with col:
            confidence = float(probs[digit]) * 100
            st.markdown(
                f"""
                <div style='text-align:center; padding:20px; border-radius:12px;
                            background:{model_colors[name]}20;
                            border: 2px solid {model_colors[name]};'>
                    <h3 style='color:{model_colors[name]};margin:0'>{name}</h3>
                    <div style='font-size:72px; font-weight:bold; line-height:1.1;
                                color:{model_colors[name]}'>{digit}</div>
                    <div style='font-size:18px; color:#555;'>{confidence:.1f}% confident</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Probability bar chart
            import plotly.graph_objects as go

            bar_colors = [
                model_colors[name] if i == digit else "#cccccc"
                for i in range(10)
            ]
            fig = go.Figure(
                go.Bar(
                    x=list(range(10)),
                    y=(probs * 100).tolist(),
                    marker_color=bar_colors,
                    text=[f"{p*100:.1f}%" for p in probs],
                    textposition="outside",
                )
            )
            fig.update_layout(
                margin=dict(t=10, b=10, l=0, r=0),
                height=200,
                xaxis=dict(title="Digit", tickmode="linear", dtick=1),
                yaxis=dict(title="%", range=[0, 110]),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Agreement banner ──────────────────────────────────────────────────────
    digits = [results[n][0] for n in results]
    if len(set(digits)) == 1:
        st.success(f"✅ All three models agree: the digit is **{digits[0]}**")
    else:
        votes = {d: digits.count(d) for d in set(digits)}
        majority = max(votes, key=votes.get)
        detail = ", ".join(
            f"{n} → **{results[n][0]}**" for n in results
        )
        st.info(f"🗳️  Models disagree — {detail}. Majority vote: **{majority}**")

elif predict_btn and arr_28x28 is None:
    st.warning("Please draw or upload a digit before predicting.")
