from image_similarity_measures.quality_metrics import rmse,ssim,psnr
from pyciede2000 import ciede2000
import cv2
import numpy as np
import nique
import psnr
import ssim

def evaluate(recovered_image,original_image,metrics):
    if metrics == 0:
        return rmse(original_image,recovered_image)
    elif metrics == 1:
        # return ssim(original_image,recovered_image)
        return ssim.ssim(original_image,recovered_image)
    elif metrics == 2:
        # return psnr(original_image,recovered_image)
        return psnr.psnr(original_image,recovered_image)
    elif metrics == 3:
        recovered_image=np.float32(recovered_image)
        original_image= np.float32(original_image)
        Lab1 = cv2.cvtColor(recovered_image, cv2.COLOR_BGR2Lab)
        L1, a1, b1 = cv2.split(Lab1)
        Lab2 = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)
        L2, a2, b2 = cv2.split(Lab2)
        res = ciede2000((np.mean(L1),np.mean(a1),np.mean(b1)),(np.mean(L2),np.mean(a2),np.mean(b2)));
        # print(b2)
        return res['delta_E_00']
    elif metrics == 4:
        nique.niqe(recovered_image)


# img=cv2.imread("5.jpg")
# img2=cv2.imread("out5.jpg")

# print(evaluate(img,img2,2))