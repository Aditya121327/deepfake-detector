import streamlit as st
from detector import detect_image
from PIL import Image

st.title("Deepfake Detector")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    if st.button("Detect"):

        result, confidence = detect_image(image)

        st.subheader("Result")
        st.write(result)

        st.subheader("Confidence")
        st.write(str(confidence) + " %")
