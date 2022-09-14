from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path
from PIL import Image
import os
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # Use VGG-16 as the architecture and ImageNet for the weight
        base_model = VGG16(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    def extract(self, img):
        # Resize the image
        img = img.resize((224, 224))
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
      
fe=FeatureExtractor()
directory = os.fsencode("/Users/shweta.grewal/hackathon/imagesearch/data/training")

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))
            
    
for img_path in sorted(os.listdir(directory)): 
    # Extract Features
    feature = fe.extract(img=Image.open(os.path.join(directory, img_path)))
    
    feature_path = "/Users/shweta.grewal/hackathon/imagesearch/data/features/feature-"+img_path.decode()+".npy"
    np.save(feature_path, feature)
    
    
import matplotlib.pyplot as plt
import pandas as pd
# Insert the image query
img = Image.open("/Users/shweta.grewal/hackathon/imagesearch/data/validation/present.jpg")
# Extract its features
query = fe.extract(img)

feature_directory="/Users/shweta.grewal/hackathon/imagesearch/data/features/"
dists=[]
for feature_path in sorted(os.listdir(feature_directory)):
# Compare euclidein distance with each given image's features
    feature = np.load(os.path.join(feature_directory, feature_path))
    dists = np.append(dists,np.linalg.norm(feature - query, axis=0))
# Extract 30 images that have lowest distance
ids = np.argsort(dists)[:3]
print(ids)

img_paths = list(absoluteFilePaths(directory))

scores = [(dists[id], img_paths[id]) for id in ids]
# Visualize the result
axes=[]
fig=plt.figure(figsize=(8,8))
for a in range(3):
    score = scores[a]
    axes.append(fig.add_subplot(5, 6, a+1))
    subplot_title=str(score[0])
    axes[-1].set_title(subplot_title)  
    plt.axis('off')
    plt.imshow(Image.open(score[1]))
fig.tight_layout()
plt.show()
