import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pyttsx3
import pandas as pd
import os
import pickle
from datetime import datetime, timedelta



st.set_page_config(page_title='Pneumonia Detection', page_icon=':lungs:')
st.title("Pneumonia Detection")

model = load_model('D:/Machine Learning DataSet/Cheast_xray/chest_xray.h5')

# Add CSS styling for the page
st.markdown(
    """
    <style>
    .stApp {
     background-image: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
    }
    .stImage {
        max-width: 100%;
        max-height:50%;
        border-radius: 20px;
        box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.15);
    }
       .prediction-table {
        margin-top: 2rem;
        border-collapse: collapse;
        width: 100%;
        font-size: 14px;
        font-family: Arial, sans-serif;
    }
    
    .prediction-table th,
    .prediction-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    
    .prediction-table th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the saved results from the file
if os.path.exists("results.pkl"):
    with open("results.pkl", "rb") as file:
        results = pickle.load(file)
else:
    results = pd.DataFrame(columns=["image_name", "result", "date"])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_name = uploaded_file.name
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img = load_img(uploaded_file, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    classes = model.predict(img_data)
    prediction = classes[0][0]
    
    threshold = 0.5  # Adjust this threshold value as needed
    
    if prediction >= threshold:
        st.write("The person in the image is NORMAL.")
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say("The person in the image is normal.")
        if not engine._inLoop:
            engine.runAndWait()
        result_text = "NORMAL"
    else:
        st.write("The person in the image is affected by PNEUMONIA.")
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say("The person in the image is affected by pneumonia.")
        if not engine._inLoop:
            engine.runAndWait()
        result_text = "PNEUMONIA"

    # Rest of the code for saving results and displaying the table...

    
    # Save the results in a dataframe
    now = datetime.now()
    date = now.strftime("%Y-%m-%d %H:%M:%S")
    results = results.append({"image_name": image_name, "result": result_text, "date": date}, ignore_index=True)

    # Filter the results for the past week
    week_ago = now - timedelta(days=7)
    filtered_results = results[results["date"] > week_ago.strftime("%Y-%m-%d %H:%M:%S")]

    # Save the filtered results as a file
    with open("results.pkl", "wb") as file:
        pickle.dump(filtered_results, file)

    # Display the results in the dashboard
    st.write("Past week's results:")
    table = st.table(filtered_results)
