import os
import integrate
import evaluation
import cv2

original_path = "dataset/data/original_images/"
training_path = "dataset/data/training_images/"

clear_image_src='out9.jpg'

list_of_metrics_of_dcp=[]
list_of_metrics_of_erc=[]

files=os.listdir(original_path)
for original_file in files:
    left_name_split=original_file[5:]
    right_name_split=left_name_split[:-4]

    training_files=os.listdir(training_path)
    for training_file in training_files:
         training_name_left=training_file[5:]
         training_name_right=training_name_left[:-8]
         if right_name_split==training_name_right:
             print(original_path+original_file)
             print(training_path+training_file)
             clear_img=cv2.imread(original_path+original_file)
             dcp_img,erc_img = integrate.getResult(training_path+training_file)
             list_of_metrics_of_dcp.append((evaluation.evaluate(dcp_img,clear_img,1),evaluation.evaluate(dcp_img,clear_img,2)))
             list_of_metrics_of_erc.append((evaluation.evaluate(erc_img,clear_img,1),evaluation.evaluate(erc_img,clear_img,2)))
             print(list_of_metrics_of_dcp,list_of_metrics_of_erc)
             exit(0)






