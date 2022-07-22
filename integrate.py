import cv2
from cv2 import COLOR_BGR2GRAY
import numpy as np
import dcp
import bcp
import union
import airlight
import guidedfilter
import recover
import evaluation
import Airlight

def getResult(img,clear_image_src):
    # print(img_src)
    kernel_size = 15
    patch_size= 60
    omega = 0.95
    epsilon = 0.1
    filter_radius = 60
    filter_regularize = 0.0001

    # img_utf8 = cv2.resize(cv2.imread(img_src), (640, 480))
    # img_hsv = cv2.cvtColor(img_utf8, cv2.COLOR_RGB2HSV)

    # # Histogram equalisation on the V-channel
    # img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

    # # convert image back from HSV to RGB
    # image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    # cv2.imshow("equalizeHist", image)
    # cv2.waitKey(0)

    # print(img_utf8)
    # img = img_utf8/255
    data = 255*img
    img_utf8=data.astype(np.uint8)
    # kernel_size=int(img_utf8.shape[1]/4)
    # patch_size = int(img_utf8.shape[1]/6)
    # filter_radius = int(img_utf8.shape[1]/10)
    # clear_img=cv2.imread(clear_image_src)/255
    # print(img)
    # b ,g ,r =cv2.split(img)
    darkchannel = dcp.getdarkChannel(img,kernel_size)
    brighchannel = bcp.getBrightChannel(img,kernel_size)

    A = airlight.airlight(img,patch_size);
    # A=Airlight.Airlight(img_utf8,"fast",kernel_size)
    A_dcp=dcp.estimateAmosphericLight(darkchannel,img)
    # A=[1,1,1]

    # tempimg = np.zeros(img.shape,dtype='float')
    # for i in range(3):
    #     tempimg[:,:,i] = img[:,:,i]/A[i]
    # brighchannel=bcp.getBrightChannel(tempimg,kernel_size)

    transmission_dcp = dcp.estimateTransmission(img,A,omega,kernel_size)
    transmission_dcp_test = dcp.estimateTransmission(img,A_dcp,omega,kernel_size)
    # noise= cv2.GaussianBlur(transmission_dcp_test, (15,15),100) 
    # mask= transmission_dcp_test - noise
    # transmission_dcp_test = transmission_dcp_test + 0.75*mask
    transmission_dcp_test=guidedfilter.guidedFilter(img_utf8,transmission_dcp_test,filter_radius,filter_regularize)
    

    # kernel = np.ones((5,5),np.uint8)
    # transmission_dcp_test=cv2.morphologyEx(transmission_dcp_test, cv2.MORPH_CLOSE, kernel)
    transmission_bcp = bcp.estimateTransmission(img,A,omega,kernel_size)
    union_transition= union.findUnion(transmission_dcp,transmission_bcp,darkchannel,brighchannel,A)
    refined_transition= guidedfilter.guidedFilter(img_utf8,union_transition,filter_radius,filter_regularize)
    # refined_transition=cv2.equalizeHist(refined_transition)
    # refined_transition=cv2.GaussianBlur(refined_transition, (11,11),10) 
    dcp_recovered_image = recover.recoverImage(img,A_dcp,transmission_dcp_test,epsilon)
    recovered_image=recover.recoverImage(img,A,refined_transition,epsilon)
    # print('darkchannel image\n',darkchannel)
    # print('brightchannel image\n',brighchannel)
    # print('transmission_dcp_test image\n',transmission_dcp_test)
    # print('transmission_dcp image\n',transmission_dcp)
    # print('transmission_bcp image\n',transmission_bcp)
    # print('union_transition image\n',union_transition)
    # print('refined image\n',refined_transition)
    # print('dcp image\n',dcp_recovered_image)
    # print('resultant image\n',recovered_image)
    # cv2.imshow("dcp",darkchannel);
    # cv2.imshow("bcp",brighchannel);
    # cv2.imshow("trans_dcp",transmission_dcp);
    # cv2.imshow("trans_bcp",transmission_bcp);
    # cv2.imshow("trans_union",union_transition);
    # cv2.imshow("trans_refined",refined_transition);
    # cv2.imshow("hazy",img)
    # cv2.imshow("recovered dcp image",dcp_recovered_image);
    # cv2.imshow("recovered image",recovered_image);
    # cv2.imshow("image",clear_img)
    # evaluation.evaluate(recovered_image,clear_img,1)
    # evaluation.evaluate(dcp_recovered_image,clear_img,1)
    # print(A)
    # noise= cv2.GaussianBlur(img, (11,11),1000) 
    # mask= img - noise
    # recovered_image = recovered_image + mask
    # kernel = np.array([[0, -1, 0],
    #                [-1, 5,-1],
    #                [0, -1, 0]])
    # image_sharp = cv2.filter2D(src=dcp_recovered_image, ddepth=-1, kernel=kernel)
    
    # temp=image_sharp
    # temp=recovered_image
    # noise= cv2.GaussianBlur(temp, (11,11),10) 
    # mask= temp - noise
    # temp = temp + mask
    # temp=dcp_recovered_image
    # temp=cv2.cvtColor(temp,COLOR_GRAY2BGR)
    # temp=image_sharp
    # cv2.anisotropicDiffusion(	temp, temp, 10)
    

    # noise= cv2.GaussianBlur(temp, (15,15),1) 
    # mask= temp - noise
    # temp = temp + 0.55*mask
    # temp=cv2.medianBlur(temp,5)

    # gray=cv2.cvtColor(temp,COLOR_)

    # grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # # abs_grad_x = cv2.convertScaleAbs(grad_x)
    # # abs_grad_y = cv2.convertScaleAbs(grad_y)

    # # grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # grad=grad_x + grad_y
    # temp=guidedfilter.guidedFilter(img_utf8,grad,filter_radius,filter_regularize)
    # # noise= cv2.GaussianBlur(grad, (15,15),100) 
    # # mask= temp - noise
    # # temp = temp + 0.5*mask
    # # temp=temp+0.25*noise

    # cv2.imshow('grad X',grad_x)
    # cv2.imshow('grad Y',grad_y)
    # cv2.imshow('Sobel Image',temp)
    # cv2.waitKey()
    

    # temp = cv2.convertScaleAbs(temp, alpha=1.5, beta=0)
    # cv2.imshow("minus",temp)
    # temp=recovered_image
    # kernel = np.ones((kernel_size,kernel_size),np.float32)
    # erosion = cv2.erode(recovered_image,kernel,iterations = 1)
    
    cv2.waitKey();
    return (dcp_recovered_image,recovered_image)


