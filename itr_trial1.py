import torch.nn as nn
import os
import string
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
import torch.optim as optim
import wandb  # Import WandB
from transformers import BertModel
from torchvision import models

data = {}

# Read the file contents
with open(r"C:\Users\chakr\OneDrive\Desktop\results_20130124.token", 'r', encoding='utf-8') as file:
    for line in file:
        # Split the line at the tab to separate the image name and sentence
        image_info, sentence = line.split('\t')
        # Extract the image name by splitting the string before the '#'
        image_name = image_info.split('#')[0]
        
        # Add the sentence to the dictionary for the corresponding image
        if image_name in data:
            data[image_name].append(sentence.strip())  # Append the sentence
        else:
            data[image_name] = [sentence.strip()]  # Create a new entry for this image

# Convert the dictionary to an array where each element is [image_name, sentences]
result = []
for image_name, sentences in data.items():
    # Join the sentences with a comma
    sentences_str = ''.join(sentences)
    result.append([image_name, sentences_str])

'''
--------------------------------------------------------------------------------------------------
'''


def preprocess_text(text):
    # Lowercase: Ensure uniform text case
    text = text.lower()
    
    # Optional: Remove punctuation (you can keep it if you think punctuation is meaningful)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace: Standardize the spacing in the text
    text = ' '.join(text.split())
    
    # Tokenize: Split text into individual words (tokens)
    tokens = word_tokenize(text)
    
    # NOTE: We are NOT removing stop words to preserve the full sentence structure for attention models
    
    # Convert tokens back to string (optional; some models prefer tokenized input)
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Function to process an array like result[0]
def preprocess_array(array):
    image_name = array[0]  # Image file name
    text = array[1]        # Text description
    preprocessed_text = preprocess_text(text)
    return [image_name, preprocessed_text]


# Apply preprocessing
preprocessed_result = [preprocess_array(item) for item in result]

final_input = preprocessed_result[0:10000]

'''
----------
'''


# Custom dataset class for image-text retrieval
class ImageTextDataset(Dataset):
    def __init__(self, image_paths, texts, tokenizer, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Tokenize the text with padding
        text = self.texts[idx]
        encoded_text = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)

        # Return image, input_ids, and attention_mask, all appropriately padded
        return image, encoded_text['input_ids'].squeeze(), encoded_text['attention_mask'].squeeze()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Example tokenizer (using BERT as an example)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocessed image paths and texts (replace with your actual preprocessed_result)
image_paths =[pair[0] for pair in final_input]  # Use actual paths

texts = [pair[1] for pair in final_input]  # Using preprocessed texts from your data


# Custom collate function to handle padding for variable-length tensors
def collate_fn(batch):
    images, input_ids, attention_masks = zip(*batch)

    # Stack images normally (all images are the same size due to transformations)
    images = torch.stack(images)

    # Pad the text tensors
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return images, input_ids, attention_masks

# Example model for image-text cross attention
class ImageTextCrossAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Pretrained ResNet model for image feature extraction
        resnet = models.resnet50(pretrained=True)
        # Remove the final fully connected layer (we'll add our own)
        self.vision_model = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection layer to map ResNet features to 256-dim vector
        self.image_projection = nn.Linear(resnet.fc.in_features, 256)
        self.dropout_image = nn.Dropout(p=0.5)  # Dropout for image features

        # Load pre-trained BERT model to get text embeddings
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Linear layer to project BERT's output embeddings to a 256-dim vector
        self.text_projection = nn.Linear(768, 256)
        self.dropout_text = nn.Dropout(p=0.5)  # Dropout for text features
    
    def forward(self, image, input_ids, attention_mask):
        # Pass image through ResNet model
        image_features = self.vision_model(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten the output
        image_features = self.image_projection(image_features)  # Project to 256-dim
        image_features = self.dropout_image(image_features)  # Apply dropout to image features
        
        # Pass text through BERT model to get text embeddings
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # Get the [CLS] token representation
        
        # Project BERT embeddings to the same dimension as image features
        text_features = self.text_projection(text_embeddings)
        text_features = self.dropout_text(text_features)  # Apply dropout to text features
        
        return image_features, text_features

# Define model, optimizer, and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImageTextCrossAttentionModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) # L2 regularization 

# Contrastive Loss (or InfoNCE Loss)
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        # Normalize the features
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        # Compute cosine similarity
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        
        # Labels for contrastive learning: Positive pairs are on the diagonal
        labels = torch.arange(logits.size(0)).to(logits.device)
        
        # Apply cross entropy loss
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

criterion = ContrastiveLoss()


# Training function with WandB logging
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        images, input_ids, attention_mask = batch
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        # Forward pass
        image_features, text_features = model(images, input_ids, attention_mask)
        
        # Calculate loss
        loss = criterion(image_features, text_features)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log loss to WandB
        wandb.log({"Training Loss": loss.item()})

    # Return average loss for the epoch
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, device, split_name="Validation"):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            images, input_ids, attention_mask = batch
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            image_features, text_features = model(images, input_ids, attention_mask)

            # Calculate loss (for evaluation)
            loss = criterion(image_features, text_features)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    
    # Log evaluation loss to WandB
    wandb.log({f"{split_name} Loss": avg_loss})
    
    return avg_loss

from sklearn.model_selection import train_test_split

# Split the data into train, validation, and test sets (80-10-10 split)
image_paths_train, image_paths_temp, texts_train, texts_temp = train_test_split(
    image_paths, texts, test_size=0.2, random_state=42
)
image_paths_val, image_paths_test, texts_val, texts_test = train_test_split(
    image_paths_temp, texts_temp, test_size=0.5, random_state=42
)

# Create datasets for each split
train_dataset = ImageTextDataset(image_paths_train, texts_train, tokenizer, transform)
val_dataset = ImageTextDataset(image_paths_val, texts_val, tokenizer, transform)
test_dataset = ImageTextDataset(image_paths_test, texts_test, tokenizer, transform)

# Create DataLoaders for each split
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)


os.chdir(r'C:\Users\chakr\Downloads\extracted_files\flickr30k-images')


wandb.init(project="image-text-retrieval", entity="chakravarty-aditya28-texas-a-m-university")  # Set your project and username
# Training loop with WandB tracking
num_epochs = 80
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Training phase
    avg_train_loss = train(model, train_dataloader, criterion, optimizer, device)
    
    # Validation phase
    avg_val_loss = evaluate(model, val_dataloader, device, split_name="Validation")
    
    # Log epoch-level information to WandB
    wandb.log({"Epoch": epoch+1, "Average Training Loss": avg_train_loss, "Average Validation Loss": avg_val_loss})

# After training, test the model on the test set
test_loss = evaluate(model, test_dataloader, device, split_name="Test")
print(f"Test Loss: {test_loss}")
wandb.finish()
