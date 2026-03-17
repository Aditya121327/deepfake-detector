import streamlit as st
from detector import detect_image
from PIL import Image

st.set_page_config(page_title="Deepfake Detector")

st.title("AI Deepfake Detection System")

st.write(
    "Upload an image to check whether it is Real or AI Generated"
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect"):

        result, confidence, reason = detect_image(image)

        if result == "Fake":
            st.error(f"Result: {result}")
        else:
            st.success(f"Result: {result}")

        st.write("Confidence:", confidence, "%")
        st.write("Reason:", reason)
