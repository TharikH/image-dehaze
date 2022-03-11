import evaluation
import cv2
import integrate

img_url="5.jpg"
clear_img_url="out5.jpg"

dcp,erc=integrate.getResult(img_url,clear_img_url)

# cv2.imshow("dcp",dcp)
# cv2.imshow("erc",erc)
print(dcp.shape)
# print(evaluation.evaluate(dcp,erc,1))
cv2.waitKey()

