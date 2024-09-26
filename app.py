from datetime import datetime
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify, session


# model import
from model.git_base_model import generate_captions_git_base  # Import from git_large_model.py
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import base64


app = Flask(__name__)



# Image caption function start 
@app.route('/imageCaptioning', methods=['POST'])
def predict_caption():
    generated_caption = None
    image = None
    image2generate = None  # Initialize image2generate
    
    image_file = request.files.get('imageFile')
    image_url = request.form.get('imageURL')
    selected_model = request.form.get('model')

    if image_file:
        image2generate = Image.open(image_file)
        image = f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
    elif image_url:
        response = requests.get(image_url)
        image2generate = Image.open(BytesIO(response.content))
        image = f"data:image/jpeg;base64,{base64.b64encode(response.content).decode('utf-8')}"
    else:
        return jsonify({'error': 'No image file or URL provided'}), 400  # Return an error if no image is provided

    if image2generate:
        generated_caption = generate_captions_git_base(image2generate)
    else:
        return jsonify({'error': 'Failed to process image'}), 500  # Return an error if image processing fails

    return jsonify({'caption': generated_caption, 'image': image})

# Image caption function end

if __name__ == "__main__":
    app.run(debug=True)
