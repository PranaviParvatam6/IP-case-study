
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


# -------------------------------
# Project configuration
# -------------------------------
MODEL_PATH = Path(
    "runs/segment/runs/fruit_segmentation/first_training/weights/best.pt"
)

st.set_page_config(
    page_title="Fruit Segmentation and Counting",
    page_icon="🍎",
    layout="wide"
)


# -------------------------------
# Load the trained model
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


# -------------------------------
# Application interface
# -------------------------------
st.title("🍎 Fruit Segmentation and Automated Counting")

st.write(
    "Upload a fruit image to segment individual fruits "
    "and estimate the total number detected."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found.")
    st.write("Expected model path:", str(MODEL_PATH.resolve()))
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, caption="Original image", use_container_width=True)

    if st.button("Detect and Count Fruits", type="primary"):
        with st.spinner("Detecting and segmenting fruits..."):
            model = load_model()

            results = model.predict(
                source=image,
                imgsz=416,
                conf=0.25,
                device="cpu",
                verbose=False
            )

            result = results[0]

            if result.masks is not None:
                fruit_count = len(result.masks.xy)
            else:
                fruit_count = 0

            annotated_image = result.plot()

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fruits Detected", fruit_count)

        with col2:
            st.metric("Image Width", f"{image.width} px")

        st.image(
            annotated_image,
            caption="Detected fruit segmentation",
            channels="BGR",
            use_container_width=True
        )

        # Download the annotated image
        annotated_pil = Image.fromarray(annotated_image[:, :, ::-1])
        from io import BytesIO

        buffer = BytesIO()
        annotated_pil.save(buffer, format="PNG")

        st.download_button(
            label="Download Annotated Image",
            data=buffer.getvalue(),
            file_name="fruit_segmentation_result.png",
            mime="image/png"
        )

st.caption(
    "Note: The displayed count is the number of fruit instances "
    "predicted by the model. Results may vary with image quality "
    "and fruit overlap."
)