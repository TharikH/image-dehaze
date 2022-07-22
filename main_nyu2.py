import os
import integrate
import evaluation
import cv2
import evaluation_store
import numpy as np
import example
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob

count = 0


original_path = "original_images/"
training_path = "training_images/"

list_of_train_files=[]

list_of_dcp_ssim=[]
list_of_dcp_psnr2=[]
list_of_dcp_psnr=[]
list_of_dcp_brisque=[]

list_of_erc_psnr=[]
list_of_erc_ssim=[]
list_of_erc_psnr2=[]
list_of_erc_brisque=[]

list_of_aod_brisque=[]
list_of_aod_psnr=[]
list_of_aod_ssim=[]
list_of_aod_psnr2=[]

list_of_dcp_aod_brisque=[]
list_of_dcp_aod_psnr=[]
list_of_dcp_aod_ssim=[]
list_of_dcp_aod_psnr2=[]

list_of_aod_dcp_brisque=[]
list_of_aod_dcp_psnr=[]
list_of_aod_dcp_ssim=[]
list_of_aod_dcp_psnr2=[]

list_of_erc_aod_brisque=[]
list_of_erc_aod_psnr=[]
list_of_erc_aod_ssim=[]
list_of_erc_aod_psnr2=[]

list_of_aod_erc_brisque=[]
list_of_aod_erc_psnr=[]
list_of_aod_erc_ssim=[]
list_of_aod_erc_psnr2=[]

list_of_erc_dcp_brisque=[]
list_of_erc_dcp_psnr=[]
list_of_erc_dcp_ssim=[]
list_of_erc_dcp_psnr2=[]

list_of_dcp_erc_brisque=[]
list_of_dcp_erc_psnr=[]
list_of_dcp_erc_ssim=[]
list_of_dcp_erc_psnr2=[]


dehaze_net = net.dehaze_net()
dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth',map_location=torch.device('cpu')))



files=os.listdir(original_path)
for original_file in files:
    if count == 5:
        break
    count+=1
    left_name_split=original_file[5:]
    right_name_split=left_name_split[:-4]

    training_files=os.listdir(training_path)
    for training_file in training_files:
         training_name_left=training_file[5:]
         training_name_right=training_name_left[:-8]
         if right_name_split==training_name_right:
            clear_img=cv2.imread(original_path+original_file)
            hazy_image=cv2.imread(training_path+training_file)


            cv2.imwrite("op/nyu2-clear-"+str(count)+".png", clear_img)
            cv2.imwrite("op/nyu2-hazy-"+str(count)+".png", hazy_image)


            ########################################dcp and erc#################################################
            data_hazy=hazy_image/255
            dcp_img,erc_img = integrate.getResult(data_hazy,clear_img)
            # cv2.imshow("dcp",dcp_img)
            # cv2.imshow("erc",erc_img)

            cv2.imwrite("op/nyu2-dcp-"+str(count)+".png", dcp_img*255)
            cv2.imwrite("op/nyu2-erc-"+str(count)+".png", erc_img*255)

            data = 255*dcp_img
            dcp_img=data.astype(np.uint8)

            data = 255*erc_img
            erc_img=data.astype(np.uint8)

            # list_of_dcp_ssim.append(evaluation.evaluate(dcp_img,clear_img,1))
            # list_of_erc_ssim.append(evaluation.evaluate(erc_img,clear_img,1))
            # list_of_dcp_psnr.append(evaluation.evaluate(dcp_img,clear_img,2))
            # list_of_erc_psnr.append(evaluation.evaluate(erc_img,clear_img,2))
            # list_of_erc_brisque.append(evaluation.evaluate(erc_img,clear_img,3))
            # list_of_dcp_brisque.append(evaluation.evaluate(dcp_img,clear_img,3))
            # list_of_erc_psnr2.append(evaluation.evaluate(erc_img,clear_img,4))
            # list_of_dcp_psnr2.append(evaluation.evaluate(dcp_img,clear_img,4))


            #####################################aodnet###########################################################

            data_hazy=hazy_image/255.0
            data_hazy = torch.from_numpy(data_hazy).float()
            data_hazy = data_hazy.permute(2,0,1)
            data_hazy = data_hazy.unsqueeze(0)
            
            clean_image = dehaze_net(data_hazy)
            res=clean_image.squeeze()
            res=res.permute(1,2,0)
            res=res.detach().numpy()

            # cv2.imshow("aodnet",res)
            cv2.imwrite("op/nyu2-aod-"+str(count)+".png", res*255)
            
            data = 255*res
            res=data.astype(np.uint8)

            # list_of_aod_ssim.append(evaluation.evaluate(res,clear_img,1))
            # list_of_aod_psnr.append(evaluation.evaluate(res,clear_img,2))
            # list_of_aod_brisque.append(evaluation.evaluate(res,clear_img,3))
            # list_of_aod_psnr2.append(evaluation.evaluate(res,clear_img,4))


            #####################################dcp+aodnet && erc+aodnet######################################################

            data_hazy=hazy_image/255
            dcp_img,erc_img = integrate.getResult(data_hazy,clear_img)

            erc_img = torch.from_numpy(erc_img).float()
            erc_img = erc_img.permute(2,0,1)
            erc_img = erc_img.unsqueeze(0)
            
            clean_image = dehaze_net(erc_img)
            erc_img=clean_image.squeeze()
            erc_img=erc_img.permute(1,2,0)
            erc_img=erc_img.detach().numpy()

            dcp_img = torch.from_numpy(dcp_img).float()
            dcp_img = dcp_img.permute(2,0,1)
            dcp_img = dcp_img.unsqueeze(0)

            clean_image = dehaze_net(dcp_img)
            dcp_img=clean_image.squeeze()
            dcp_img=dcp_img.permute(1,2,0)
            dcp_img=dcp_img.detach().numpy()

            # cv2.imshow("dcp_aod",dcp_img)
            # cv2.imshow("erc_aod",erc_img)
            cv2.imwrite("op/nyu2-dcp_aod-"+str(count)+".png", dcp_img*255)
            cv2.imwrite("op/nyu2-erc_aod-"+str(count)+".png", erc_img*255)
            

            data = 255*dcp_img
            dcp_img=data.astype(np.uint8)

            data = 255*erc_img
            erc_img=data.astype(np.uint8)

            # list_of_dcp_aod_ssim.append(evaluation.evaluate(dcp_img,clear_img,1))
            # list_of_erc_aod_ssim.append(evaluation.evaluate(erc_img,clear_img,1))
            # list_of_dcp_aod_psnr.append(evaluation.evaluate(dcp_img,clear_img,2))
            # list_of_erc_aod_psnr.append(evaluation.evaluate(erc_img,clear_img,2))
            # list_of_erc_aod_brisque.append(evaluation.evaluate(erc_img,clear_img,3))
            # list_of_dcp_aod_brisque.append(evaluation.evaluate(dcp_img,clear_img,3))
            # list_of_erc_aod_psnr2.append(evaluation.evaluate(erc_img,clear_img,4))
            # list_of_dcp_aod_psnr2.append(evaluation.evaluate(dcp_img,clear_img,4))
            

            ################################# aod+dcp && aod+erc ###########################

            data_hazy=hazy_image/255.0
            data_hazy = torch.from_numpy(data_hazy).float()
            data_hazy = data_hazy.permute(2,0,1)
            data_hazy = data_hazy.unsqueeze(0)
            
            clean_image = dehaze_net(data_hazy)
            res=clean_image.squeeze()
            res=res.permute(1,2,0)
            res=res.detach().numpy()

            dcp_img,erc_img = integrate.getResult(res,clear_img)

            # cv2.imshow("aod_dcp",dcp_img)
            # cv2.imshow("aod_erc",erc_img)
            cv2.imwrite("op/nyu2-aod_dcp-"+str(count)+".png", dcp_img*255)
            cv2.imwrite("op/nyu2-aod_erc-"+str(count)+".png", erc_img*255)
            

            data = 255*dcp_img
            dcp_img=data.astype(np.uint8)

            data = 255*erc_img
            erc_img=data.astype(np.uint8)

            # list_of_aod_dcp_ssim.append(evaluation.evaluate(dcp_img,clear_img,1))
            # list_of_aod_erc_ssim.append(evaluation.evaluate(erc_img,clear_img,1))
            # list_of_aod_dcp_psnr.append(evaluation.evaluate(dcp_img,clear_img,2))
            # list_of_aod_erc_psnr.append(evaluation.evaluate(erc_img,clear_img,2))
            # list_of_aod_erc_brisque.append(evaluation.evaluate(erc_img,clear_img,3))
            # list_of_aod_dcp_brisque.append(evaluation.evaluate(dcp_img,clear_img,3))
            # list_of_aod_erc_psnr2.append(evaluation.evaluate(erc_img,clear_img,4))
            # list_of_aod_dcp_psnr2.append(evaluation.evaluate(dcp_img,clear_img,4))

            
            ################################# dcp+bcp && bcp+dcp ###########################

            data_hazy=hazy_image/255
            dcp_img_temp,erc_img_temp = integrate.getResult(data_hazy,clear_img)
            temp,erc_img = integrate.getResult(dcp_img_temp,clear_img)
            dcp_img,temp = integrate.getResult(erc_img_temp,clear_img)
            # cv2.imshow("erc_dcp",dcp_img)
            # cv2.imshow("dcp_erc",erc_img)

            cv2.imwrite("op/nyu2-erc_dcp-"+str(count)+".png", dcp_img*255)
            cv2.imwrite("op/nyu2-dcp_erc-"+str(count)+".png", erc_img*255)
            
            break
            data = 255*dcp_img
            dcp_img=data.astype(np.uint8)

            data = 255*erc_img
            erc_img=data.astype(np.uint8)

            # list_of_erc_dcp_ssim.append(evaluation.evaluate(dcp_img,clear_img,1))
            # list_of_dcp_erc_ssim.append(evaluation.evaluate(erc_img,clear_img,1))
            # list_of_erc_dcp_psnr.append(evaluation.evaluate(dcp_img,clear_img,2))
            # list_of_dcp_erc_psnr.append(evaluation.evaluate(erc_img,clear_img,2))
            # list_of_dcp_erc_brisque.append(evaluation.evaluate(erc_img,clear_img,3))
            # list_of_erc_dcp_brisque.append(evaluation.evaluate(dcp_img,clear_img,3))
            # list_of_dcp_erc_psnr2.append(evaluation.evaluate(erc_img,clear_img,4))
            # list_of_erc_dcp_psnr2.append(evaluation.evaluate(dcp_img,clear_img,4))



            # res=example.contextregularization(hazy)
            # print(evaluation.evaluate(res,clear_img,2))
            # res=(dcp_img+res/255)/2

            # noise= cv2.GaussianBlur(res, (11,11),1) 
            # mask= res - noise
            # res = res + mask
            # res=dcp_img
            # res=erc_img
            # data = 255 * res
            # res = data.astype(np.uint8)

            # res1=dcp_img
            # data = 255*res1
            # res1=data.astype(np.uint8)

            # cv2.convertScaleAbs(res,res)




            #learning based  
            # data_hazy = (np.asarray(hazy_image)/255.0)
            # data_hazy=hazy_image/255.0
            # # data_hazy=dcp_img

            # data_hazy = torch.from_numpy(data_hazy).float()
            # data_hazy = data_hazy.permute(2,0,1)
            # data_hazy = data_hazy.unsqueeze(0)
            
            # clean_image = dehaze_net(data_hazy)
            # res1=clean_image.squeeze()
            # res1=res1.permute(1,2,0)
            # res1=res1.detach().numpy()

            # data = 255*res1
            # res1=data.astype(np.uint8)

            # dcp_img,erc_img = integrate.getResult(hazy_image,clear_img)
            # print("hello")

            # data = 255*dcp_img
            # res1=data.astype(np.uint8)

            # data = 255*erc_img
            # res=data.astype(np.uint8)





            # list_of_dcp_ssim.append(evaluation.evaluate(res1,clear_img,1))
            # list_of_erc_ssim.append(evaluation.evaluate(res,clear_img,1))
            # list_of_dcp_psnr.append(evaluation.evaluate(res1,clear_img,2))
            # list_of_erc_psnr.append(evaluation.evaluate(res,clear_img,2))
            # list_of_erc_brisque.append(evaluation.evaluate(res,clear_img,3))
            # list_of_dcp_brisque.append(evaluation.evaluate(res1,clear_img,3))
            # print(list_of_dcp_ssim,list_of_erc_ssim,list_of_dcp_psnr,list_of_erc_psnr)

print("erc=","psnr ",np.mean(list_of_erc_psnr),"ssim ",np.mean(list_of_erc_ssim),"brisque ",np.mean(list_of_erc_brisque),"psnr2",np.mean(list_of_erc_psnr2))
print("dcp=","psnr ",np.mean(list_of_dcp_psnr),"ssim ",np.mean(list_of_dcp_ssim),"brisque ",np.mean(list_of_dcp_brisque),"psnr2",np.mean(list_of_dcp_psnr2))
print("aod=","psnr ",np.mean(list_of_aod_psnr),"ssim ",np.mean(list_of_aod_ssim),"brisque ",np.mean(list_of_aod_brisque),"psnr2",np.mean(list_of_aod_psnr2))
print("aod_dcp=","psnr ",np.mean(list_of_aod_dcp_psnr),"ssim ",np.mean(list_of_aod_dcp_ssim),"brisque ",np.mean(list_of_aod_dcp_brisque),"psnr2",np.mean(list_of_aod_dcp_psnr2))
print("aod_erc=","psnr ",np.mean(list_of_aod_erc_psnr),"ssim ",np.mean(list_of_aod_erc_ssim),"brisque ",np.mean(list_of_aod_erc_brisque),"psnr2",np.mean(list_of_aod_erc_psnr2))
print("dcp_aod=","psnr ",np.mean(list_of_dcp_aod_psnr),"ssim ",np.mean(list_of_dcp_aod_ssim),"brisque ",np.mean(list_of_dcp_aod_brisque),"psnr2",np.mean(list_of_dcp_aod_psnr2))
print("erc_aod=","psnr ",np.mean(list_of_erc_aod_psnr),"ssim ",np.mean(list_of_erc_aod_ssim),"brisque ",np.mean(list_of_erc_aod_brisque),"psnr2",np.mean(list_of_erc_aod_psnr2))
print("dcp_erc=","psnr ",np.mean(list_of_dcp_erc_psnr),"ssim ",np.mean(list_of_dcp_erc_ssim),"brisque ",np.mean(list_of_dcp_erc_brisque),"psnr2",np.mean(list_of_dcp_erc_psnr2))
print("erc_dcp=","psnr ",np.mean(list_of_erc_dcp_psnr),"ssim ",np.mean(list_of_erc_dcp_ssim),"brisque ",np.mean(list_of_erc_dcp_brisque),"psnr2",np.mean(list_of_erc_dcp_psnr2))
# evaluation_store.store(list_of_dcp_psnr,list_of_dcp_ssim,list_of_erc_psnr,list_of_erc_ssim,list_of_train_files)






