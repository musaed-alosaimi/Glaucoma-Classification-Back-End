from fastapi import FastAPI, File, UploadFile
from typing import Annotated
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification, Swinv2ForImageClassification
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # Convert Greyscale image to RGBscale
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),          # Convert to tensor
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])

processor = AutoImageProcessor.from_pretrained("./AI_model")
model = Swinv2ForImageClassification.from_pretrained("./AI_model")

@app.get("/test")
async def test():

    try:
        # Load the image
        image_path = "./resources/images/example.jpg"
        image = Image.open(image_path)
        # Apply the transformations
        image_tensor = transform(image)
        # Add batch dimension (models expect a batch of inputs)
        image_tensor = image_tensor.unsqueeze(0)  # Shape becomes [1, C, H, W]
        inputs = processor(image_tensor, return_tensors="pt", do_rescale=False)

        with torch.no_grad():
            logits = model(**inputs).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        print("The result is: "+model.config.id2label[predicted_label])

        return {'message': "The result is: "+model.config.id2label[predicted_label]}
    except:
        print("An exception occurred")

    return {'message': "An issue has been occured."}

@app.post("/predict")
async def read_item(file: UploadFile = File(...)):
    filename = file.filename
    image = await file.read()

    # Optionally, save the file to the server
    with open(f"./uploaded_files/eye-image.jpg", "wb") as f:
        f.write(image)

    try:
        # Load the image
        image_path = f"./uploaded_files/eye-image.jpg"
        image = Image.open(image_path)
        # Apply the transformations
        image_tensor = transform(image)
        # Add batch dimension (models expect a batch of inputs)
        image_tensor = image_tensor.unsqueeze(0)  # Shape becomes [1, C, H, W]
        inputs = processor(image_tensor, return_tensors="pt", do_rescale=False)

        with torch.no_grad():
            logits = model(**inputs).logits

        # model predicts one of the 1000 ImageNet classes
        predicted_label = logits.argmax(-1).item()
        print("The result is: "+model.config.id2label[predicted_label])

        result_output = model.config.id2label[predicted_label]

        return {'result': {'data': result_output, 'error': None}}
    except:
        print("An exception occurred")

    return {'resutl': {'data': None, 'error': "An issue has been occured."}}
