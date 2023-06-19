# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 00:48:13 2023

@author: 14055
"""

import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


''' preprocess data from a PDF journal publication'''

import PyPDF2
import re
import string
import nltk
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

def preprocess_pdf(pdf_file_path):
    # Load PDF file
    with open(pdf_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        
        # Extract text from each page
        raw_text = ''
        for page_number in range(num_pages):
            page = pdf_reader.pages[page_number]
            raw_text += page.extract_text()
            
        # Text cleaning
        raw_text = re.sub('\n', ' ', raw_text)  # Remove line breaks
        raw_text = re.sub(' +', ' ', raw_text)  # Remove multiple spaces
        
        # Tokenization
        tokens = word_tokenize(raw_text)
        
        # Stopword removal
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.lower() not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Sentence segmentation
        sentences = sent_tokenize(raw_text)
        
        return tokens, sentences

# Usage example
pdf_file = 'ps1.pdf'  # Replace with the path to your PDF file
tokens, sentences = preprocess_pdf(pdf_file)

# Print the preprocessed tokens
print(tokens)

# Print the preprocessed sentences
print(sentences)



def preprocess_pdf(file_path):
    # Open the PDF file
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        content = ""
        
        # Extract the text from each page
        for page in range(num_pages):
            content += pdf_reader.pages[page].extract_text()
        
        # Split the content into sentences
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", content)
        
        # Filter out sentences with excessive numbers, special characters, or 'doi'
        filtered_sentences = []
        for sentence in sentences:
            # Remove sentences with more than 3 numbers, excessive special characters, or 'doi'
            if re.search(r"\d{3,}|\W{5,}|doi", sentence, re.IGNORECASE):
                continue
            filtered_sentences.append(sentence)
        
        # Join the filtered sentences
        preprocessed_text = " ".join(filtered_sentences)
        
        return preprocessed_text

# Example usage
pdf_file_path = "ps1.pdf"  # Update with your PDF file path
preprocessed_text = preprocess_pdf(pdf_file_path)

from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, LineByLineTextDataset
from transformers import get_linear_schedule_with_warmup, AdamW


# instantiate GPT2 tokenizer, byte-level encoding
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# add special tokens that otherwise all share the same id
tokenizer.add_special_tokens({'bos_token': '<bos>',
                              'eos_token': '<eos>',
                              'pad_token': '<pad>'})

# instantiate model GPT2 transformer with a language modeling head on top
model = GPT2LMHeadModel.from_pretrained('distilgpt2').cuda()  # to GPU


tokens = tokenizer.encode_plus(preprocessed_text, add_special_tokens=True, padding='longest', max_length=512, truncation=True, return_tensors="pt")

# Fine-tuning the model
model.train()

# Set the input sequence
input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

optimizer = AdamW(model.parameters(), lr=1e-5)  
num_training_steps = 100  # Replace with the desired number of training steps
progress_bar = tqdm(total=num_training_steps, desc="Training Progress")

# Move the tensors to the same device
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
labels = input_ids.to(device)  # Assuming labels are the same as input_ids

# Fine-tune the model
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)



for step in range(num_training_steps):
    # Perform a forward pass and calculate loss
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    # Perform backward pass and update model parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Update the progress bar
    progress_bar.update(1)
    progress_bar.set_postfix({"Loss": loss.item()})

# Close the progress bar
progress_bar.close()