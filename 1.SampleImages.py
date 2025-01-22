import cv2
from matplotlib import pyplot as plt
import os

# Function to read and log image loading
def load_image(path):
    if not os.path.exists(path):
        print(f"Error: File {path} does not exist.")
        return None
    image = cv2.imread(path)
    if image is None:
        print(f"Error: Unable to read image {path}.")
    return image

# Read images with error handling
images = []
image_paths = [
    './DB1_B/AVe/101_1.tif', './DB1_B/AVe/101_2.tif', 
    './DB1_B/AVe/101_3.tif', './DB1_B/AVe/101_4.tif',
    './DB1_B/AVe/101_5.tif', './DB1_B/AVe/101_6.tif', 
    './DB1_B/AVe/101_7.tif', './DB1_B/AVe/101_8.tif'
]

for path in image_paths:
    img = load_image(path)
    if img is not None:
        images.append(img)

if len(images) < len(image_paths):
    print("Error: Some images could not be loaded.")
else:
    rows, columns = 4, 4
    fig = plt.figure(figsize=(8, 8))

    for i, img in enumerate(images[:8], 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Image {i}")

    plt.show()

# Repeat similar steps for the second set of images

image_paths = [
    './DB1_B/A-Ve/102_1.tif', './DB1_B/A-Ve/102_2.tif', 
    './DB1_B/A-Ve/102_3.tif', './DB1_B/A-Ve/102_4.tif',
    './DB1_B/A-Ve/102_5.tif', './DB1_B/A-Ve/102_6.tif', 
    './DB1_B/A-Ve/102_7.tif', './DB1_B/A-Ve/102_8.tif'
]

images = []

for path in image_paths:
    img = load_image(path)
    if img is not None:
        images.append(img)

if len(images) < len(image_paths):
    print("Error: Some images could not be loaded.")
else:
    fig = plt.figure(figsize=(8, 8))

    for i, img in enumerate(images[:8], 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Image {i}")

    plt.show()
