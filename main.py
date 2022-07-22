import os
import integrate
import evaluation
import cv2
import evaluation_store
import numpy as np

original_path = "dataset/data/original_images/"
training_path = "dataset/data/training_images/"

list_of_dcp_ssim=[]
list_of_erc_ssim=[]
list_of_dcp_psnr=[]
list_of_erc_psnr=[]
list_of_train_files=[]

files=os.listdir(original_path)
for original_file in files:
    left_name_split=original_file[5:]
    right_name_split=left_name_split[:-4]

    training_files=os.listdir(training_path)
    for training_file in training_files:
         training_name_left=training_file[5:]
         training_name_right=training_name_left[:-8]
         if right_name_split==training_name_right:
             clear_img=cv2.imread(original_path+original_file)
             img = cv2.imread(training_path+training_file)
             dcp_img,erc_img = integrate.getResult(img,clear_img)
             erc_img=erc_img*255
             erc_img=erc_img.astype(np.float64)
            #  list_of_dcp_ssim.append(evaluation.evaluate(dcp_img,clear_img,1))
            #  list_of_erc_ssim.append(evaluation.evaluate(dcp_img,clear_img,1))
             list_of_dcp_psnr.append(evaluation.evaluate(erc_img,clear_img,2))
             list_of_erc_psnr.append(evaluation.evaluate(erc_img,clear_img,2))
             print(list_of_dcp_psnr,list_of_erc_psnr)
            #  list_of_train_files.append(training_file)
            #  cv2.imshow("dcp",dcp_img)
            #  cv2.imshow("bcp",erc_img)
            #  cv2.waitKey();

evaluation_store.store(list_of_dcp_psnr,list_of_dcp_ssim,list_of_erc_psnr,list_of_erc_ssim,list_of_train_files)






