from image_similarity_measures.quality_metrics import rmse,ssim,psnr

import cv2


def evaluate(recovered_image,original_image,metrics):
    if metrics == 0:
        return rmse(original_image,recovered_image)
    elif metrics == 1:
        return ssim(original_image,recovered_image)
    elif metrics == 2:
        return psnr(original_image,recovered_image)

# img=cv2.imread("5.jpg")
# img2=cv2.imread("out5.jpg")

# print(evaluate(img,img2,2))