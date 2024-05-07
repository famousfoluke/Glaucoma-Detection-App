import streamlit as st
import tensorflow as tf

model = tf.keras.models.load_model('my_model_Final.h5')

st.title("Glaucoma Early Detection App")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_image:
    image = tf.image.decode_image(uploaded_image.read(), channels=3)
    image = tf.image.resize(image, (240, 240))  # Resize to match your model input size
    image = tf.expand_dims(image, axis=0)
    prediction = model.predict(image)
    st.write(f"Predicted class: {prediction.argmax()}")


