# needed to remove irrelavant warning
import os
import io
import json
import requests
import base64
from io import BytesIO
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel 

# Load the processor and model
git_processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
git_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

# gpu usage if available,might cause error lol 
device = "cuda" if torch.cuda.is_available() else "cpu"
git_model.to(device)

def generate_caption(processor, model, image, tokenizer=None):
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
    if tokenizer is not None:
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    else:
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return generated_caption

def generate_captions_git_base(image):
    caption_git_large = generate_caption(git_processor, git_model, image)
    return caption_git_large


#fast api 
app = FastAPI(title="Johan's API", description="fyp api")
    
class ImageCaption(BaseModel):
    caption:str

# redirect to docs 
@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs")


@app.post('/imageCaptioning', response_model=ImageCaption)
# def predict(file: UploadFile = File(...)):
#     contents= file.file.read()
#     image = Image.open(io.BytesIO(contents))
#     generated_caption = generate_captions_git_base(image)
#     return JSONResponse(content={'caption': generated_caption, 'image': image})
#fast api

def predict_caption(file: UploadFile = File(...)):
    generated_caption = None
    image = None
    image2generate = None  # Initialize image2generate
    
    image_file = file.file.read()

    if image_file:
        image2generate = Image.open(io.BytesIO(image_file))
        # image = f"data:image/jpeg;base64,{base64.b64encode(image_file.file.file.read()).decode('utf-8')}"
    
    if image2generate:
        generated_caption = generate_captions_git_base(image2generate)
    else:
        return JSONResponse({'error': 'Failed to process image'}), 500  # Return an error if image processing fails

    return JSONResponse(content={'caption': generated_caption})


# # test model
# image = Image.open("backend/model/boy-eat.jpg") 
# print(generate_captions_git_base(image=image))
