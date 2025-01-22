import sys
import os
import cv2
import imutils
from scipy.linalg import norm
from scipy import sum, average
from matplotlib.pyplot import imread

def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

dim = [100, 100]

file1 = open("fingerprint_train.csv", "w")
st1 = "label,"

for i in range(1, 10001):
    st1 += 'pixel' + str(i) + ","

st1 += 'Group,'
st1 = st1[:-1]
file1.write(st1 + '\n')

def process_images(directory, label):
    files = os.listdir(directory)
    cnt = 0

    for f in files:
        cnt += 1
        img = imread(os.path.join(directory, f), 0)
        img = cv2.resize(img, (100, 100))

        print(cnt)
        print('Reading File' + f)

        st1 = str(label) + ','
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                st1 += str(int(img[i, j])) + ','

        print()
        st1 += directory.split("/")[-1] + ", "
        st1 = st1[:-1]
        file1.write(st1 + '\n')

process_images('./DB1_B/AVe', 1)
process_images('./DB1_B/A-Ve', 0)
process_images('./DB1_B/BVe', 1)
process_images('./DB1_B/B-Ve', 0)
process_images('./DB1_B/OVe', 1)
process_images('./DB1_B/O-Ve', 0)
process_images('./DB1_B/ABVe', 1)
process_images('./DB1_B/AB-Ve', 0)

file1.close()
