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


-----------------------redeploy with requirements.txt in place----------------
#deployment code

import sagemaker
from sagemaker.pytorch import PyTorchModel

# Initialize a SageMaker session
sagemaker_session = sagemaker.Session()

# Get the execution role
role = sagemaker.get_execution_role()

# Specify the S3 path where the model artifacts are stored
model_artifact = 's3://your-bucket/path/to/model.tar.gz'

# Create a PyTorchModel object
pytorch_model = PyTorchModel(model_data=model_artifact,
                             role=role,
                             framework_version='1.8.1',  # or any other compatible version
                             py_version='py3',
                             entry_point='inference.py',  # Path to your inference script
                             source_dir='source')  # Path to the directory with your inference script and dependencies

# Deploy the model to an endpoint
predictor = pytorch_model.deploy(initial_instance_count=1, instance_type='ml.m5.large')


# testing the endpoint

import numpy as np
import json
import cv2
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Initialize the predictor (assuming the predictor object from deployment is reused)
predictor = Predictor(endpoint_name='your-endpoint-name', 
                      serializer=JSONSerializer(), 
                      deserializer=JSONDeserializer(),
                      sagemaker_session=sagemaker.Session())

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



