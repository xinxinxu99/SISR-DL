### MCNet-DL builds on the Mixed 2D/3D Convolutional Network (MCNet) originally proposed by Li et al. for hyperspectral image super-resolution.  
#- Original MCNet repository: https://github.com/qianngli/MCNet/tree/master  
#- Reference: Li, Q., Wang, Q., & Li, X. (2020). *Mixed 2D/3D convolutional network for hyperspectral image super-resolution*. Remote Sensing, 12(10), 1660.


import os
import torch
import torch.nn as nn
from Option import opt
from Model import MCNet
import numpy as np
import scipy.io as scio  
from Metrics import PSNR, SAM, ergas

    
def main():
    Dataset_name = opt.datasetName
    scale = opt.upscale_factor
    LR_HSI_mat = "./SISR-DL/HSI/" + Dataset_name + "_X" + str(scale) + "/" + Dataset_name + "_X" + str(scale) + "_end6MV.mat"
    out_path = './SISR-DL/Results/'+Dataset_name+'_X'+str(scale)+'/'
    output_filename = 'Output_'+Dataset_name+'_X'+str(scale)+'_SISR_MCNet_DL.mat'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
                    
    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    model = MCNet(opt)

    if opt.cuda:
        model = nn.DataParallel(model).cuda()    
        
    checkpoint  = torch.load(opt.model_name)

    model.load_state_dict(checkpoint["model"])  
    model.eval()  
      
        
    mat = scio.loadmat(LR_HSI_mat)
    HSI_LR = mat['LR']
    Abund_LR = mat['Abund_LR'] 
    Abund_LR = np.expand_dims(Abund_LR, axis=0)
    endmembers_lr = mat['endmembers']
    HSI_HR = mat['HR']
    hyperLR = Abund_LR
    endmembers_lr = endmembers_lr	        	
    input = torch.from_numpy(hyperLR).float()
    if opt.cuda:
        input = input.cuda()  
    with torch.no_grad():
        output = model(input)     
    SR = output.cpu().data[0].numpy().astype(np.float64)     
    SR = SR
    if not out_path.endswith('/'):
        out_path += '/'

    
    file_path = os.path.join(out_path, output_filename)
    print("Saving to file: {}".format(file_path))
    Abund_SR = SR
    HSI_SR = np.zeros((Abund_SR.shape[0], Abund_SR.shape[1], endmembers_lr.shape[1]))
    for i in range(6):
        HSI_SR += np.expand_dims(Abund_SR[:,:,i], axis=2) * np.expand_dims(endmembers_lr[i,:], axis=0)
    print("=====PSNR HSI:{:.3f}=====SAM HSI:{:.3f}====ERGAS HSI:{:.3f}".format(PSNR(HSI_HR,HSI_SR), SAM(HSI_HR,HSI_SR), ergas(HSI_HR, HSI_SR)))
 
    scio.savemat(file_path, {'Abund_SR':SR, 'Abund_LR':Abund_LR, 'hr':HSI_HR, 'sr':HSI_SR, 'lr':HSI_LR, 'endmembers_lr':endmembers_lr})
if __name__ == "__main__":
    main()
    
