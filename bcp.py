import cv2
import numpy as np

def getBrightChannel(img,kernel_size):
    b,g,r = cv2.split(img)
    brightchannel = cv2.max(b,cv2.max(g,r))
    size=(kernel_size,kernel_size)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape,size)
    brightchannel = cv2.dilate(brightchannel,kernel)
    return brightchannel

def estimateTransmission(img,A,omega,kernel_size):
    tempimg = np.zeros(img.shape,dtype='float')
    for i in range(3):
        tempimg[:,:,i] = img[:,:,i]/A[i]
    transmission = (getBrightChannel(tempimg,kernel_size) - 1 )/(1/np.max(A) - 1)
    return transmission