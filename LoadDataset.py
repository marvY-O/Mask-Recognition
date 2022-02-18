import cv2
import numpy as np
import os

path = "./without_mask//"
print("Loading dataset of people without mask....")
data = []
for imgPath in os.listdir(path):
    face = cv2.imread(path+imgPath)
    face = cv2.resize(face, (50, 50))
    if (len(data) < 1250):
        data.append(face)
    else:
        break

np.save('NoMask.npy', data)
print("Dataset of people without mask saved!")


path = "./with_mask//"
print("Loading dataset of people with mask....")
dataMask = []
for imgPath in os.listdir(path):
    face = cv2.imread(path+imgPath)
    face = cv2.resize(face, (50, 50))
    if (len(dataMask) < 1250):
        dataMask.append(face)
    else:
        break

np.save('Mask.npy', dataMask)
print("Dataset of people with mask saved!")
print("Run main.py now.")