from image_similarity_measures.quality_metrics import rmse,ssim,psnr

def evaluate(recovered_image,original_image,metrics):
    if metrics == 0:
        print(rmse(original_image,recovered_image))
    elif metrics == 1:
        print(ssim(original_image,recovered_image))
    elif metrics == 2:
        print(psnr(original_image,recovered_image))