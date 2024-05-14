import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import fiftyone as fo
import fiftyone.brain as fob

def load_images(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            images.append(img_array)
            filenames.append(filename)
    return images, filenames

# Delete the old dataset if it exists
dataset_name = "image_clusters"
if fo.dataset_exists(dataset_name):
    fo.delete_dataset(dataset_name)

# Load images
folder = r"C:\Users\14055\Desktop\51_sample"
images, filenames = load_images(folder)

# Convert images to embeddings (flattened pixel values)
embeddings = [img.flatten() for img in images]
embeddings = np.array(embeddings)

# Define the number of clusters
num_clusters = 5  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(embeddings)

# Get cluster labels
cluster_labels = kmeans.labels_

# Create a new FiftyOne dataset
dataset = fo.Dataset(dataset_name)

# Add samples to the dataset
for idx, (img_array, filename) in enumerate(zip(images, filenames)):
    img_path = os.path.join(folder, filename)
    sample = fo.Sample(filepath=img_path)
    # Add embedding and cluster_id as sample fields
    sample["embedding"] = embeddings[idx].tolist()
    sample["cluster_id"] = fo.Classification(label=str(cluster_labels[idx]))
    dataset.add_sample(sample)

# Compute visualization for embeddings
fob.compute_visualization(dataset, embeddings="embedding", brain_key="embeddings_viz")

# Launch the FiftyOne app and visualize
session = fo.launch_app(dataset)
session.wait()
