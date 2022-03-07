import cv2
import numpy as np
def guidedFilter(image,input,filter_radius,regularize):
    #step 1
    guidance_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    guidance_image = guidance_image / 255

    filter_size=(filter_radius,filter_radius)
    # print(filter_size)
    I_mean = cv2.blur(guidance_image,filter_size)
    p_mean = cv2.blur(input,filter_size)
    I_corr = cv2.blur(guidance_image * guidance_image,filter_size)
    Ip_corr = cv2.blur(guidance_image * input,filter_size)
    # step 2
    I_var = I_corr - I_mean * I_mean
    Ip_cov = Ip_corr - I_mean * p_mean
    # step 3
    a = Ip_cov/(I_var + regularize)
    b = p_mean - a*I_mean
    # step 4
    a_mean = cv2.blur(a,filter_size)
    b_mean = cv2.blur(b,filter_size)
    # step 5
    output = a_mean * guidance_image + b_mean
    return output
