import cv2
import numpy as np

def getdarkChannel(img,kernel_size):
    b,g,r = cv2.split(img)
    minchannel = cv2.min(b,cv2.min(g,r))
    size=(kernel_size,kernel_size)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape,size)
    darkchannel = cv2.erode(minchannel,kernel)
    return darkchannel

def estimateAmosphericLight(darkchannel,img):
    h,w = darkchannel.shape
    tempimg = darkchannel.reshape(h*w)
    size=tempimg.shape[0]
    indexes = tempimg.argsort()[::-1][:int(size*(0.1))]
    tempimg2 = img.reshape(size,3)
    print(tempimg2)
    max = np.sum(tempimg2[indexes[0]])
    maxindex = indexes[0]
    for i in range(indexes.shape[0]):
        temp = np.sum(tempimg2[indexes[i]])
        if max < temp:
            max = temp
            maxindex = indexes[i]
    return tempimg2[maxindex]

def estimateTransmission(img,A,omega,kernel_size):
    tempimg = np.zeros(img.shape,dtype='float')
    for i in range(3):
        tempimg[:,:,i] = img[:,:,i]/A[i]
    transmission = 1 - omega * getdarkChannel(tempimg,kernel_size)
    return transmission
