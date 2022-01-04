import numpy as np
import cv2

def recoverImage(img,A,transmission,epsilon):
    trans = cv2.max(transmission,epsilon)
    res = np.zeros(img.shape,dtype=img.dtype)  
    for i in range(3):
        res[:,:,i] = (img[:,:,i] - A[i]) / trans + A[i]
    return res