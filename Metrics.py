import math
import numpy as np

def PSNR(pred, gt):
      
    valid = gt - pred
    rmse = math.sqrt(np.mean(valid ** 2))   
    
    if rmse == 0:
        return 100
    psnr = 20 * math.log10(1.0 / rmse)
    return psnr
	
    
def SAM(pred, gt):
    eps = 2.2204e-16
    pred[np.where(pred==0)] = eps
    gt[np.where(gt==0)] = eps 
      
    nom = sum(pred*gt)
    denom1 = sum(pred*pred)**0.5
    denom2 = sum(gt*gt)**0.5
    sam = np.real(np.arccos(nom.astype(np.float32)/(denom1*denom2+eps)))
    sam[np.isnan(sam)]=0     
    sam_sum = np.mean(sam)*180/np.pi   	       
    return  sam_sum

def ergas(reference, estimate, r=4):

    B = reference.shape[2]
    mse = np.mean((reference - estimate) ** 2, axis=(0, 1))
    mean = np.mean(reference, axis=(0, 1))
    return 100 / r * np.sqrt(np.sum(mse / mean ** 2) / B)

