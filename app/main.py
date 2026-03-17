import streamlit as st

st.title("Deepfake Detector")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image")
