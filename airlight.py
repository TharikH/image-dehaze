import cv2
import numpy as np

# def airlight(img,kernel_size):
#     b,g,r = cv2.split(img)
#     size=(kernel_size,kernel_size)
#     shape = cv2.MORPH_RECT
#     kernel = cv2.getStructuringElement(shape,size)
#     r_max = cv2.dilate(r,kernel)
#     g_max = cv2.dilate(g,kernel)
#     b_max = cv2.dilate(b,kernel)
#     r_avg=np.sum(r_max)/np.size(r_max)
#     g_avg=np.sum(g_max)/np.size(g_max)
#     b_avg=np.sum(b_max)/np.size(b_max)
#     A=[b_avg,g_avg,r_avg]
#     return A
def airlight(img,kernel_size):
    height=img.shape[0]
    width=img.shape[1]
    b_max=[]
    r_max=[]
    g_max=[]
    b,g,r = cv2.split(img)
    A=[]
    for i in range(0,height,kernel_size):
        for j in range(0,width,kernel_size):
            temp_b=0
            temp_r=0
            temp_g=0
            for k in range(kernel_size):
                if i+k >= height:
                    break
                for l in range(kernel_size):
                    if j+l >= width:
                        break
                    temp_b=temp_b if temp_b>=b[i+k][j+l] else b[i+k][j+l]
                    temp_r=temp_r if temp_r>=r[i+k][j+l] else r[i+k][j+l]
                    temp_g=temp_g if temp_g>=g[i+k][j+l] else g[i+k][j+l]
            b_max.append(temp_b)
            g_max.append(temp_g)
            r_max.append(temp_r)
    A.append(sum(b_max)/len(b_max))
    A.append(sum(g_max)/len(g_max))
    A.append(sum(r_max)/len(r_max))
    return A
