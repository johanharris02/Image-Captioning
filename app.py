from flask import Flask, render_template, request
from git_large_model import generate_captions_large  # Assuming this is your caption generation module
from PIL import Image
import requests
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predictCaption():
    generatedCaption = None
    image = None

    if request.method == 'POST':
        imageFile = request.files.get('imageFile')
        image_url = request.form.get('imageURL')

        if imageFile:
            # Save the uploaded image to a temporary path
            image_path = "./images/" + imageFile.filename
            imageFile.save(image_path)
            # Open the image for processing
            image2generate = Image.open(image_path)
            # Encode the image in base64 for display
            with open(image_path, "rb") as img_file:
                image = f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
        elif image_url:
            response = requests.get(image_url)
            image2generate = Image.open(BytesIO(response.content))
            # Encode the image in base64 for display
            image = f"data:image/jpeg;base64,{base64.b64encode(response.content).decode('utf-8')}"

        # Generate caption for the image
        generatedCaption = generate_captions_large(image2generate)

    return render_template("index.html", caption=generatedCaption, image=image)

if __name__ == "__main__":
    app.run(port=3000, debug=True)
