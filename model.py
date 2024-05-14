import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('my_model_Final.h5')

# Define ImageDataGenerator for preprocessing
test_datagen = ImageDataGenerator(
    shear_range=0.5,
    zoom_range=0.005,
)

# Define a mapping from prediction output to descriptive labels
prediction_labels = {
    0: "Normal",
    1: "Early Glaucoma",
    2: "Advanced Glaucoma"
}

# Streamlit App
st.title("Glaucoma Early Detection App")

st.write("This app helps in detecting glaucoma in eye images by classifying it into Normal, Early Glaucoma and Advanced Glaucoma class. Please upload only retina fundus images")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_image:
    # Read and preprocess the image
    image = Image.open(uploaded_image)
    image = image.resize((240, 240))  # Resize to match your model input size

    # Convert the image to a NumPy array and expand the dimensions
    image_np = np.array(image)
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension

    # Apply the preprocessing (shear and zoom) to the image
    preprocessed_image = test_datagen.standardize(image_np)  # Apply standardization

    # Make a prediction using the preprocessed image
    prediction = model.predict(preprocessed_image)

    # Plot the accuracy of the prediction on a bar chart
    fig, ax = plt.subplots()
    predicted_labels = [prediction_labels[i] for i in range(len(prediction[0]))]
    ax.bar(predicted_labels, prediction[0])
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Accuracy')
    st.pyplot(fig)

    # Get the class with the highest probability
    predicted_class = prediction.argmax()

    # Get the label associated with the predicted class
    predicted_label = prediction_labels[predicted_class]

    # Display the prediction result with the corresponding label
    st.write(f"Predicted result: {predicted_label}")


