import streamlit as st
from detector import detect_image
from PIL import Image

st.set_page_config(
    page_title="Deepfake Detector",
    layout="wide"
)

st.title("AI Deepfake Detection System")

st.write(
    "Upload image to check authenticity using AI-based analysis"
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file)

    with col1:
        st.image(image, caption="Input")

    if st.button("Detect"):

        with st.spinner("Running model..."):

            result, confidence, reason, heatmap = detect_image(image)

        with col2:

            if result == "Fake":
                st.error(result)
            else:
                st.success(result)

            st.progress(confidence / 100)

            st.write("Confidence:", confidence, "%")

            st.write(reason)

        st.subheader("Analysis Map")

        st.image(heatmap)
