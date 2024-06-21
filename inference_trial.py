import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import json
import numpy as np

# Import your model architecture
from vegan_model import VegAnnModel

def model_fn(model_dir):
    # Load the model architecture
    model = VegAnnModel("Unet", "resnet34", in_channels=3, out_classes=1)
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def input_fn(request_body, request_content_type='application/json'):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        image = np.array(input_data['image'])
        preprocess_input = smp.encoders.get_preprocessing_fn('resnet34', pretrained='imagenet')
        image = preprocess_input(image)
        image = image.astype('float32')
        inputs = torch.tensor(image)
        inputs = inputs.permute(2, 0, 1)
        inputs = inputs[None, :, :, :]
        return inputs
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    with torch.no_grad():
        logits = model(input_data)
        pr_mask = logits.sigmoid()
        pred = (pr_mask > 0.5).numpy().astype(np.uint8)
    return pred

def output_fn(prediction, content_type='application/json'):
    if content_type == 'application/json':
        response_body = json.dumps(prediction.tolist())
        return response_body
    else:
        raise ValueError(f"Unsupported content type: {content_type}")







-------------------------------------

import numpy as np
import json
import cv2

# Function to preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')
    return img

# Path to your input image
image_path = 'path/to/your/input/image.jpg'
input_image = preprocess_image(image_path)

# Create the request payload
request_payload = {
    'image': input_image.tolist()  # Convert the numpy array to list
}

# Send the request to the endpoint and get the response
response = predictor.predict(request_payload)

# Convert the response back to numpy array
prediction = np.array(json.loads(response))

# Print the prediction result
print("Prediction result:", prediction)
