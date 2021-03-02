import numpy as np
import random
import os
import cv2
from tqdm import tqdm
import pickle

# Main directory for the images
DATADIR = "New folder"

# Folder name of each action
CATEGORIES = ["Diving", "Golf Swing", "Kicking", "Lifting", "Riding Horse", "Running", "SkateBoarding", "Swing-Bench",
              "Swing-Side", "Walking"]

# Image dimensions
IMG_SIZE = 360
IMG_SIZE2 = 240

training_data = []

def create_training_data():
    for category in CATEGORIES: # Goes through the folders of each category

        path = os.path.join(DATADIR, category)  # Creates a path to the category
        class_num = CATEGORIES.index(category)  # Gets the classification(0-9). 0 = Diving, 1 = Golf, etc

        for img in tqdm(os.listdir(path)): # Iterates over each image per category (Diving, Golf, Kicking, etc)
                try:
                    img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # Converts to array
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE2))  # Resize to normalize data size
                    training_data.append([new_array, class_num])  # Add this to our training_data
                except Exception as e:
                    pass


create_training_data()

print()
print(len(training_data))

random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

# Creating X and y to hold the data that will be used in the model
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE2, 1)
my_y = np.zeros((len(y),10))
for i in range(len(y)):
    my_y[i][y[i]] = 1
y = my_y
del(my_y)

# Using pickle to save the data to be used in the model so I don't have to process the data every time
# I make changes to the model
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

