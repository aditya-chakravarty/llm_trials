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
