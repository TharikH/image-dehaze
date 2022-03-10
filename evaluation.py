from image_similarity_measures.quality_metrics import rmse,ssim,psnr

def evaluate(recovered_image,original_image,metrics):
    if metrics == 0:
        return rmse(original_image,recovered_image)
    elif metrics == 1:
        return ssim(original_image,recovered_image)
    elif metrics == 2:
        return psnr(original_image,recovered_image)