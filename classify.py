import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tempfile

# -----------------------
# CONFIGURATION
# -----------------------
MODEL_PATH = "mobilenetv2_fish.h5"  # Change to your desired model
IMG_SIZE = (224, 224)  # Match your training size
CLASS_LABELS = ['Fish_Class_1', 'Fish_Class_2', 'Fish_Class_3']  # Update with your class names

# Load model once
@st.cache_resource
def load_fish_model():
    return load_model(MODEL_PATH)

model = load_fish_model()

# -----------------------
# STREAMLIT UI
# -----------------------
st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish and I'll tell you its category with confidence scores.")

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    # Load & preprocess image
    img = image.load_img(temp_file.name, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence_scores = preds[0] * 100

    # Display uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prediction result
    st.subheader(f"Prediction: **{CLASS_LABELS[pred_class]}**")
    st.write("### Confidence Scores:")
    for label, score in zip(CLASS_LABELS, confidence_scores):
        st.write(f"- {label}: {score:.2f}%")

    # Optional: Display as bar chart
    st.bar_chart(dict(zip(CLASS_LABELS, confidence_scores)))
