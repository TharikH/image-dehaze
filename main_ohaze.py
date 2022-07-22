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


original_path = "O-HAZE/# O-HAZY NTIRE 2018/GT/"
training_path = "O-HAZE/# O-HAZY NTIRE 2018/hazy/"

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
    
    
    name_split=original_file[0:2]
    training_file=name_split+'_outdoor_hazy.jpg'
    print(training_file)

    
    
    
                
    clear_img=cv2.imread(original_path+original_file)
    clear_img=cv2.resize(clear_img, (640, 480))
    hazy_image=cv2.imread(training_path+training_file)
    hazy_image=cv2.resize(hazy_image, (640, 480))

    cv2.imwrite("op/ohaze-clear-"+str(count)+".png", clear_img)
    cv2.imwrite("op/ohaze-hazy-"+str(count)+".png", hazy_image)


    ########################################dcp and erc#################################################
    data_hazy=hazy_image/255
    dcp_img,erc_img = integrate.getResult(data_hazy,clear_img)
    # cv2.imshow("dcp",dcp_img)
    # cv2.imshow("erc",erc_img)

    cv2.imwrite("op/ohaze-dcp-"+str(count)+".png", dcp_img*255)
    cv2.imwrite("op/ohaze-erc-"+str(count)+".png", erc_img*255)

    data = 255*dcp_img
    dcp_img=data.astype(np.uint8)

    data = 255*erc_img
    erc_img=data.astype(np.uint8)

    list_of_dcp_ssim.append(evaluation.evaluate(dcp_img,clear_img,1))
    list_of_erc_ssim.append(evaluation.evaluate(erc_img,clear_img,1))
    list_of_dcp_psnr.append(evaluation.evaluate(dcp_img,clear_img,2))
    list_of_erc_psnr.append(evaluation.evaluate(erc_img,clear_img,2))
    list_of_erc_brisque.append(evaluation.evaluate(erc_img,clear_img,3))
    list_of_dcp_brisque.append(evaluation.evaluate(dcp_img,clear_img,3))
    list_of_erc_psnr2.append(evaluation.evaluate(erc_img,clear_img,4))
    list_of_dcp_psnr2.append(evaluation.evaluate(dcp_img,clear_img,4))


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
    cv2.imwrite("op/ohaze-aod-"+str(count)+".png", res*255)
    
    data = 255*res
    res=data.astype(np.uint8)

    list_of_aod_ssim.append(evaluation.evaluate(res,clear_img,1))
    list_of_aod_psnr.append(evaluation.evaluate(res,clear_img,2))
    list_of_aod_brisque.append(evaluation.evaluate(res,clear_img,3))
    list_of_aod_psnr2.append(evaluation.evaluate(res,clear_img,4))


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
    cv2.imwrite("op/ohaze-dcp_aod-"+str(count)+".png", dcp_img*255)
    cv2.imwrite("op/ohaze-erc_aod-"+str(count)+".png", erc_img*255)
    

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
    cv2.imwrite("op/ohaze-aod_dcp-"+str(count)+".png", dcp_img*255)
    cv2.imwrite("op/ohaze-aod_erc-"+str(count)+".png", erc_img*255)
    

    data = 255*dcp_img
    dcp_img=data.astype(np.uint8)

    data = 255*erc_img
    erc_img=data.astype(np.uint8)

    list_of_aod_dcp_ssim.append(evaluation.evaluate(dcp_img,clear_img,1))
    list_of_aod_erc_ssim.append(evaluation.evaluate(erc_img,clear_img,1))
    list_of_aod_dcp_psnr.append(evaluation.evaluate(dcp_img,clear_img,2))
    list_of_aod_erc_psnr.append(evaluation.evaluate(erc_img,clear_img,2))
    list_of_aod_erc_brisque.append(evaluation.evaluate(erc_img,clear_img,3))
    list_of_aod_dcp_brisque.append(evaluation.evaluate(dcp_img,clear_img,3))
    list_of_aod_erc_psnr2.append(evaluation.evaluate(erc_img,clear_img,4))
    list_of_aod_dcp_psnr2.append(evaluation.evaluate(dcp_img,clear_img,4))

    
    ################################# dcp+bcp && bcp+dcp ###########################

    data_hazy=hazy_image/255
    dcp_img_temp,erc_img_temp = integrate.getResult(data_hazy,clear_img)
    temp,erc_img = integrate.getResult(dcp_img_temp,clear_img)
    dcp_img,temp = integrate.getResult(erc_img_temp,clear_img)
    # cv2.imshow("erc_dcp",dcp_img)
    # cv2.imshow("dcp_erc",erc_img)

    cv2.imwrite("op/ohaze-erc_dcp-"+str(count)+".png", dcp_img*255)
    cv2.imwrite("op/ohaze-dcp_erc-"+str(count)+".png", erc_img*255)
    

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


    #dcp and erc
    # hazy_image=hazy_image/255
    # dcp_img,erc_img = integrate.getResult(hazy_image,clear_img)
    # # cv2.imshow("dcp",dcp_img)
    # # cv2.imshow("erc",erc_img)

    # data = 255*dcp_img
    # dcp_img=data.astype(np.uint8)

    # data = 255*erc_img
    # erc_img=data.astype(np.uint8)

    # list_of_dcp_ssim.append(evaluation.evaluate(dcp_img,clear_img,1))
    # list_of_erc_ssim.append(evaluation.evaluate(erc_img,clear_img,1))
    # list_of_dcp_psnr.append(evaluation.evaluate(dcp_img,clear_img,2))
    # list_of_erc_psnr.append(evaluation.evaluate(erc_img,clear_img,2))
    # list_of_erc_brisque.append(evaluation.evaluate(erc_img,clear_img,3))
    # list_of_dcp_brisque.append(evaluation.evaluate(dcp_img,clear_img,3))
    # list_of_erc_psnr2.append(evaluation.evaluate(erc_img,clear_img,4))
    # list_of_dcp_psnr2.append(evaluation.evaluate(dcp_img,clear_img,4))
    # hazy=hazy_image
    # hazy_image=hazy_image/255
    # dcp_img,erc_img = integrate.getResult(hazy_image,clear_img)

    # #res=example.contextregularization(hazy)
    # # print(evaluation.evaluate(res,clear_img,2))
    # # res=(dcp_img+res/255)/2

    # # noise= cv2.GaussianBlur(res, (11,11),1) 
    # # mask= res - noise
    # # res = res + mask
    # # res=dcp_img
    # res=erc_img
    # data = 255 * res
    # res = data.astype(np.uint8)

    # res1=dcp_img
    # data = 255*res1
    # res1=data.astype(np.uint8)
    
    # data_hazy=hazy_image/255.0
    # data_hazy=dcp_img

    # data_hazy = torch.from_numpy(data_hazy).float()
    # data_hazy = data_hazy.permute(2,0,1)
    # data_hazy = data_hazy.unsqueeze(0)
    
    # clean_image = dehaze_net(data_hazy)
    # res1=clean_image.squeeze()
    # res1=res1.permute(1,2,0)
    # res1=res1.detach().numpy()

    # data = 255*res1
    # res1=data.astype(np.uint8)

    # res1,erc_img = integrate.getResult(res1,clear_img)
    # data = 255*res1
    # res1=data.astype(np.uint8)

    # print("hello")

    # list_of_dcp_ssim.append(evaluation.evaluate(res1,clear_img,1))
    # # list_of_erc_ssim.append(evaluation.evaluate(res,clear_img,1))
    # list_of_dcp_psnr.append(evaluation.evaluate(res1,clear_img,2))
    # # list_of_erc_psnr.append(evaluation.evaluate(res,clear_img,2))
    # # list_of_erc_brisque.append(evaluation.evaluate(res,clear_img,3))
    # list_of_dcp_brisque.append(evaluation.evaluate(res1,clear_img,3))
    # # list_of_train_files.append(training_file)
    

    # # # list_of_dcp_ssim.append(evaluation.evaluate(dcp_img,clear_img,1))
    # # list_of_erc_ssim.append(evaluation.evaluate(erc_img,clear_img,1))
    # # # list_of_dcp_psnr.append(evaluation.evaluate(dcp_img,clear_img,2))
    # # list_of_erc_psnr.append(evaluation.evaluate(erc_img,clear_img,2))
    # # list_of_erc_brisque.append(evaluation.evaluate(erc_img,clear_img,3))
    # # list_of_train_files.append(training_file)

    # cv2.imshow("dcp",cv2.resize(dcp_img, (960, 540)) )
    # # cv2.imshow("hello",res)
    # # cv2.imshow("bcp",erc_img)
    # cv2.imshow("bcp",res)
    # # cv2.imshow("clear",clear_img)
    # # cv2.imshow("hazy",hazy_image)
    # # print(list_of_dcp_psnr)
    # # print(list_of_erc_psnr)
    # # print(list_of_erc_brisque)
    cv2.waitKey();

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

