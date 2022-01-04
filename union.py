import cv2
import numpy as np
def findUnion(transmission_dark,transmission_bright,darkchannel,brightchannel,A):
    res = np.zeros(transmission_dark.shape,dtype=transmission_dark.dtype)
    row,col=np.shape(transmission_bright)
    A_max=max(A)
    for i in range(row):
        for j in range(col):
            if(brightchannel[i][j] > A_max):
                res[i][j] = transmission_bright[i][j]
            else:
                res[i][j] = transmission_dark[i][j]
    
    return res