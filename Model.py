### MCNet-DL builds on the Mixed 2D/3D Convolutional Network (MCNet) originally proposed by Li et al. for hyperspectral image super-resolution.  
#- Original MCNet repository: https://github.com/qianngli/MCNet/tree/master  
#- Reference: Li, Q., Wang, Q., & Li, X. (2020). *Mixed 2D/3D convolutional network for hyperspectral image super-resolution*. Remote Sensing, 12(10), 1660.

import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F


class BasicConv3d(nn.Module):
    def __init__(self, wn, in_channel, out_channel, kernel_size, stride, padding=(0,0,0)):
        super(BasicConv3d, self).__init__()
        self.conv = wn(nn.Conv3d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
  
        x = self.conv(x)
        x = F.relu(x)  
        
        return x

class S3Dblock(nn.Module):
    def __init__(self, wn, n_feats):
        super(S3Dblock, self).__init__()

        self.conv = nn.Sequential(
            BasicConv3d(wn, n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            BasicConv3d(wn, n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))
        )            
       
    def forward(self, x): 
    	   	
        return self.conv(x)

def _to_4d_tensor(x, depth_stride=None):
    x = x.transpose(0, 2) 
    if depth_stride:
        x = x[::depth_stride]  
    depth = x.size()[0]
    x = x.permute(2, 0, 1, 3, 4)  
    x = torch.split(x, 1, dim=0) 
    x = torch.cat(x, 1) 
    x = x.squeeze(0)  
    return x, depth


def _to_5d_tensor(x, depth):
    x = torch.split(x, depth)  
    x = torch.stack(x, dim=0)  
    x = x.transpose(1, 2) 
    return x
    
    
class Block(nn.Module):
    def __init__(self, wn, n_feats, n_conv):
        super(Block, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        
        Block1 = []  
        for i in range(n_conv):
            Block1.append(S3Dblock(wn, n_feats)) 
        self.Block1 = nn.Sequential(*Block1)         

        Block2 = []  
        for i in range(n_conv):
            Block2.append(S3Dblock(wn, n_feats)) 
        self.Block2 = nn.Sequential(*Block2) 
        
        Block3 = []  
        for i in range(n_conv):
            Block3.append(S3Dblock(wn, n_feats)) 
        self.Block3 = nn.Sequential(*Block3) 
        
        self.reduceF = BasicConv3d(wn, n_feats*3, n_feats, kernel_size=1, stride=1)                                                            
        self.Conv = S3Dblock(wn, n_feats)
        self.gamma = nn.Parameter(torch.ones(3))   
         
        conv1 = []   
        conv1.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        conv1.append(self.relu)
        conv1.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1))))         
        self.conv1 = nn.Sequential(*conv1)           

        conv2 = []   
        conv2.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        conv2.append(self.relu)
        conv2.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1))))         
        self.conv2 = nn.Sequential(*conv2)  
        
        conv3 = []   
        conv3.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        conv3.append(self.relu)
        conv3.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1))))         
        self.conv3 = nn.Sequential(*conv3)          
                 
        
                                                          
    def forward(self, x): 
        
        res = x
        x1 = self.Block1(x) + x 
        x2 = self.Block2(x1) + x1         
        x3 = self.Block3(x2) + x2     

        x1, depth = _to_4d_tensor(x1, depth_stride=1)  
        x1 = self.conv1(x1)       
        x1 = _to_5d_tensor(x1, depth)  
                             
        x2, depth = _to_4d_tensor(x2, depth_stride=1)  
        x2 = self.conv2(x2)       
        x2 = _to_5d_tensor(x2, depth)         
   
                     
        x3, depth = _to_4d_tensor(x3, depth_stride=1)  
        x3 = self.conv3(x3)       
        x3 = _to_5d_tensor(x3, depth)  
                
        x = torch.cat([self.gamma[0]*x1, self.gamma[1]*x2, self.gamma[2]*x3], 1)                 
        x = self.reduceF(x) 
        x = F.relu(x)  
        
        x = x + res        
        
        
        x = self.Conv(x)                                                                                                               
        return x  
                                                                                                                        
                        
class MCNet(nn.Module):
    def __init__(self, args):
        super(MCNet, self).__init__()
        
        scale = args.upscale_factor
        n_colors = args.n_bands
        n_feats = args.n_feats          
        n_conv = args.n_conv
        kernel_size = 3
        band_mean = (0.06203818627965988, 0.08258559209040113, 0.290879074175476, 0.1385205837284263, 0.0998182729195922, 0.23439452824911916, 0) # Urban Abund lr end6MV
        #band_mean = [0.06100607699544718, 0.24623535730268548, 0.21211722223899243, 0.23508256322035834, 0.24977904511495408, 0.24810896231125532, 0] # PaviaU Abund lr end6MV
        #band_mean = (0.060286377599859185, 0.2909972484642961, 0.045824342467546166, 0.09082010577745958, 0.07013696019724813, 0.05140507208771323, 0) # Chikusei Abund lr end6MV
        
        wn = lambda x: torch.nn.utils.parametrizations.weight_norm(x)
        self.band_mean = torch.autograd.Variable(torch.FloatTensor(band_mean)).view([1, 1, 1,n_colors])
                                     
        self.head = wn(nn.Conv3d(1, n_feats, kernel_size, padding=kernel_size//2))        
               
        self.SSRM1 = Block(wn, n_feats, n_conv)              
        self.SSRM2 = Block(wn, n_feats, n_conv) 
        self.SSRM3 = Block(wn, n_feats, n_conv)           
        self.SSRM4 = Block(wn, n_feats, n_conv)  
                                                
        tail = []
        tail.append(wn(nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(2+scale,2+scale,3), stride=(scale,scale,1), padding=(1,1,1))))         
        tail.append(wn(nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size//2)))  
        self.tail = nn.Sequential(*tail)                                                                                 

    def forward(self, x):
        x = x - self.band_mean.cuda()  
        x = x.unsqueeze(1)
        T = self.head(x) 
        x = self.SSRM1(T)
        x = torch.add(x, T) 
        x = self.SSRM2(x)
        x = torch.add(x, T)     
        x = self.SSRM3(x)
        x = torch.add(x, T)                  
        x = self.SSRM4(x)
        x = torch.add(x, T) 
        x = self.tail(x)    
        x = x.squeeze(1) 
        x = x + self.band_mean.cuda()  
        
        return x  