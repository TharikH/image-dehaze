import cv2
import numpy as np
import dcp
import bcp
import union
import airlight
import guidedfilter
import recover
import evaluation

def getResult(img_src,clear_image_src):
    # print(img_src)
    kernel_size = 251
    patch_size= 10
    omega = 0.95
    epsilon = 0.1
    filter_radius = 6
    filter_regularize = 0.0001

    img_utf8 = cv2.imread(img_src)
    # print(img_utf8)
    img = img_utf8/255
    # kernel_size=img_utf8.shape[1]
    clear_img=cv2.imread(clear_image_src)/255
    # print(img)
    b ,g ,r =cv2.split(img)
    darkchannel = dcp.getdarkChannel(img,kernel_size)
    brighchannel = bcp.getBrightChannel(img,kernel_size)

    A = airlight.airlight(img,patch_size);
    A_dcp=dcp.estimateAmosphericLight(darkchannel,img)
    # A=[0.8,0.8,0.8]
    transmission_dcp = dcp.estimateTransmission(img,A,omega,kernel_size)
    transmission_dcp_test = guidedfilter.guidedFilter(img_utf8,dcp.estimateTransmission(img,A_dcp,omega,kernel_size),filter_radius,filter_regularize)
    transmission_bcp = bcp.estimateTransmission(img,A,omega,kernel_size)
    union_transition= union.findUnion(transmission_dcp,transmission_bcp,darkchannel,brighchannel,A)
    refined_transition= guidedfilter.guidedFilter(img_utf8,union_transition,filter_radius,filter_regularize)
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
    # cv2.imshow("recovered dcp image",dcp_recovered_image);
    # cv2.imshow("recovered image",recovered_image);
    # cv2.imshow("image",clear_img)
    evaluation.evaluate(recovered_image,clear_img,1)
    evaluation.evaluate(dcp_recovered_image,clear_img,1)
    print(A)
    cv2.waitKey();
    return (dcp_recovered_image,recovered_image)


