import numpy as np
from scipy import io as sio
import random
from scipy.io import  savemat
from scipy.ndimage import gaussian_filter, zoom
import os
from Option import opt

def in_leaf(X0, Y0, Lx, Ly, angle_deg, X, Y):
    angle_rad = np.deg2rad(angle_deg)
    return ((X-X0)*np.cos(angle_rad) + (Y-Y0)*np.sin(angle_rad) < Lx/2) & ((X-X0)*np.cos(angle_rad) + (Y-Y0)*np.sin(angle_rad) > -Lx/2) & ((Y-Y0)*np.cos(angle_rad) - (X-X0)*np.sin(angle_rad) < Ly/2) & ((Y-Y0)*np.cos(angle_rad) - (X-X0)*np.sin(angle_rad) > -Ly/2)


def pick_abund_value(Abund):
    x_rand = random.randint(0, Abund.shape[0] - 1)
    y_rand = random.randint(0, Abund.shape[1] - 1)
    return Abund[x_rand, y_rand, :]

def creat_abund(height, weight, num_endmembers, Xmin, Xmax, Ymin, Ymax):
    abund = np.zeros((height, weight, num_endmembers))
    mapping2D = np.ones((height, weight))
    limit_leaf = 0
    X, Y = np.meshgrid(np.arange(height), np.arange(weight), indexing='ij')
    
    while np.sum(mapping2D) > 0 and limit_leaf < 1000:
        X0 = np.random.randint(0, height)
        Y0 = np.random.randint(0, weight)
        Lx = np.random.randint(Xmin, Xmax)
        Ly = np.random.randint(Ymin, Ymax)
        angle_deg = np.random.randint(45, 90)
        limit_leaf += 1
        random_abund_val = pick_abund_value(Abund_GT_LR)
        for endmember in range(num_endmembers):
            leaf_val = random_abund_val[endmember]
            mask = (mapping2D == 1) & in_leaf(X0, Y0, Lx, Ly, angle_deg, X, Y)
            abund[mask, endmember] = leaf_val
        mapping2D[mask] = 0
        print("uncovered pixels = ", np.sum(mapping2D),end='\r')
    return abund


def generate_gaussian_noise(psnr_db, Lx, Ly, Lz):
    I_max = 1.0 
    sigmaNoise = (I_max ** 2) / (10 ** (psnr_db / 10))
    noise_HSI = np.random.randn(Lx, Ly, Lz) * np.sqrt(sigmaNoise)
    return noise_HSI, sigmaNoise

def calc_noise_Abond(noise_HSI, endmembers):
    end_pinv = np.linalg.pinv(endmembers) 
    noise_Abond = np.tensordot(noise_HSI, end_pinv, axes=([2],[0]))
    return noise_Abond

scale = opt.upscale_factor # default 4
dataset_name = opt.datasetName #Urban

path_HSI_LR =  "./SISR-DL/HSI/" + dataset_name + "_X" + str(scale) + "/" + dataset_name + "_X" + str(scale) + "_end6MV.mat" # Default : "./SISR-DL/HSI/Urban/Urban_X4_end6MV.mat"
data_HSI = sio.loadmat(path_HSI_LR)
endmembers = data_HSI['endmembers'] #shape(M,B)
HSI_LR = data_HSI['LR'] #shape(h,w,B)
Abund_GT_LR = data_HSI['Abund_LR'] #shape(h,w,M)
pre_path_dataset_DL = "./SISR-DL/Dataset/" + dataset_name + "/X" + str(scale) + "/"
pre_path_DL_Train = pre_path_dataset_DL + 'Train/'
pre_path_DL_Valid = pre_path_dataset_DL + 'Valid/'
if not os.path.exists(pre_path_DL_Train):
    os.makedirs(pre_path_DL_Train)
if not os.path.exists(pre_path_DL_Valid):
    os.makedirs(pre_path_DL_Valid)

    
height = 2  * Abund_GT_LR.shape[0] *scale
weight = 2 * Abund_GT_LR.shape[1] *scale
num_endmembers = Abund_GT_LR.shape[2]
Xmin =  2 * scale * 2
Xmax =  2 * scale * 30
Ymin = 2 * scale * 2
Ymax = 2 * scale * 30
sig = (2,2,0)
sigma4 = (4,4,0)
lambda_exp = 1 / 5
psnr_max = 60
num_iter = 5000
psnr_samples = np.abs(psnr_max - np.random.exponential(scale=1/lambda_exp, size=num_iter)) #distribution of PSNR ~ PSNR_max - Exp(lambda)
for i in range(num_iter):  
    print('Generating training dataset DL n°: ', i, end='\r')
    if i % 2 ==0:

        noise_HSI, sigmaNoise = generate_gaussian_noise(psnr_samples[i], HSI_LR.shape[0], HSI_LR.shape[1], HSI_LR.shape[2])
        noise_Abond = calc_noise_Abond(noise_HSI, endmembers)
    else:
        noise_Abond = np.zeros((HSI_LR.shape[0], HSI_LR.shape[1], HSI_LR.shape[2]))
        sigmaNoise = 0

    abund = creat_abund(height, weight, num_endmembers, Xmin, Xmax, Ymin, Ymax)
    abundHR = gaussian_filter(abund, sigma = sig)
    abundHR = zoom(abundHR, (0.5, 0.5, 1), order=1)
    abundLR = gaussian_filter(abundHR, sigma = sigma4, truncate= 3.0)
    abundLR = zoom(abundLR, (1/scale, 1/scale, 1), order=3)
    abundLR += noise_Abond
    NoiseMap = np.ones((abundLR.shape[0], abundLR.shape[1], 1)) * sigmaNoise
    abundLR = np.concatenate((abundLR, NoiseMap), axis=2)
    savemat(pre_path_DL_Train + f'/DL_{num_endmembers}_{i+1}.mat', {'lr': abundLR, 'hr' : abundHR})

num_iter = 500
psnr_samples = np.abs(psnr_max - np.random.exponential(scale=1/lambda_exp, size=num_iter))
for i in range(num_iter):
    print('Generating validation dataset DL n°: ', i, end='\r')
    if i % 2 ==0:
        noise_HSI, sigmaNoise = generate_gaussian_noise(psnr_samples[i], HSI_LR.shape[0], HSI_LR.shape[1], HSI_LR.shape[2])
        noise_Abond = calc_noise_Abond(noise_HSI, endmembers)
    else:
        noise_Abond = np.zeros((HSI_LR.shape[0], HSI_LR.shape[1], HSI_LR.shape[2]))
        sigmaNoise = 0

    abund = creat_abund(height, weight, num_endmembers, Xmin, Xmax, Ymin, Ymax)
    abundHR = gaussian_filter(abund, sigma = sig)#, truncate= 3.0)
    abundHR = zoom(abundHR, (0.5, 0.5, 1), order=1)
    abundLR = gaussian_filter(abundHR, sigma = sigma4, truncate= 3.0)
    abundLR = zoom(abundLR, (1/scale, 1/scale, 1), order=3)
    abundLR += noise_Abond
    NoiseMap = np.ones((abundLR.shape[0], abundLR.shape[1], 1)) * sigmaNoise
    abundLR = np.concatenate((abundLR, NoiseMap), axis=2)
    savemat(pre_path_DL_Valid + f'/DL_{num_endmembers}_{i+1}.mat', {'lr': abundLR, 'hr' : abundHR})