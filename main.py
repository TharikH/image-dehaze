import cv2
import numpy as np
import dcp
import bcp
import union
import airlight
import guidedfilter
import recover

img_src = '5.jpg'
kernel_size = 11
omega = 0.95
epsilon = 0.1
filter_radius = 60
filter_regularize = 0.0001



img_utf8 = cv2.imread(img_src)
img = img_utf8/255
darkchannel = dcp.getdarkChannel(img,kernel_size)
brighchannel = bcp.getBrightChannel(img,kernel_size)

# A = airlight.airlight();
A_dcp=dcp.estimateAmosphericLight(darkchannel,img)
A=[0.8,0.8,0.8]
transmission_dcp = dcp.estimateTransmission(img,A,omega,kernel_size)
transmission_bcp = bcp.estimateTransmission(img,A,omega,kernel_size)
union_transition= union.findUnion(transmission_dcp,transmission_bcp,darkchannel,brighchannel,A)
refined_transition= guidedfilter.guidedFilter(img_utf8,union_transition,filter_radius,filter_regularize)
cv2.imshow("trans_dcp",transmission_dcp);
cv2.imshow("trans_bcp",transmission_bcp);
cv2.imshow("trans_union",union_transition);
cv2.imshow("trans_refined",refined_transition);
# dcp_recovered_image = recover.recoverImage(img,A_dcp,transmission_dcp,epsilon)
# recovered_image=recover.recoverImage(img,A,refined_transition,epsilon)
# cv2.imshow("recovered dcp image",dcp_recovered_image);
# cv2.imshow("recovered image",recovered_image);
cv2.waitKey();
