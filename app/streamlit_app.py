import base64
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
from util import set_background

def predict(model, image_data):
    classes = ['Common_rust','Gray_leaf_spot', 'Healthy', 'Northen_Leaf_Blight']
    size = (299, 299)
    image_data = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image_array = np.asarray(image_data)
    image_array = image_array/255.0
    image = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image)
    predicted_class = classes[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])), 2)
    return predicted_class, confidence

def main():
    st.set_page_config(
        page_title="Corn Leaf Disease Classification",
        page_icon=":plant-doctor:"
    )
    set_background('/Users/dipit/Image Data/Corn_leaf/bg/corn_field.png')
    st.title("Corn Leaf Image Classifier")
    print("")
    file = st.file_uploader("Please upload a Corn leaf file", type=['jpg', 'jpeg', 'png'])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    @st.cache_resource()
    def load_model():
        model = keras.models.load_model('model/CornLeafInception.h5')
        return model
    with st.spinner("Model is being loaded ..."):
        model = load_model()

    if file is None:
        st.text("Please Upload an image file")
    else:
       image = Image.open(file)
       st.image(image, use_column_width=True)
       predicted_class, confidence = predict(model, image)
       st.write(f"The image likely shows {predicted_class.lower()} with confidence of {confidence}%")
       print(f"The image likely shows {predicted_class.lower()} with confidence of {confidence}%")

if __name__ == "__main__":
    main()