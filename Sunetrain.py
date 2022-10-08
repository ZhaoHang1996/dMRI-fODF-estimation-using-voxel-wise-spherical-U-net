import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from SunetBase import Database, ModelBase
from SunetNetBase import Unet_sub4


class mySunetModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super(mySunetModel, self).__init__(*args, **kwargs)
        
    def running(self, batchData):
        
        self.inputsData = batchData['indata']
        
        self.outSHC = self.network(self.inputsData)
        
        self.loss.valueCounting({'SHCloss': (self.outSHC, batchData['SHClabel'].float())})
        
        


def main():
    
    parser = argparse.ArgumentParser(description="device and epoch setting.")
    
    parser.add_argument('--device', '-d', default='cuda:0')
    parser.add_argument('--totalEpochs', '-e', default=200)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    network = Unet_sub4(1,1)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    
    model = mySunetModel(network = network,
                         device = device,
                         batchSize = 1,
                         savePath = "/data/zh/EXP_save/sunet/",
                         task = ['SHC'])
    
    model.dataPathSetting(indata= '/data/zh/path/hcp_dODF_b1000_60_Lmax8_SphVal_Ico_sub4_2562_withEro3mask_flatten_path.txt',
                          SHClabel= '/data/zh/path/hcp_mcsd_FOD_wm_0123_SphVal_Ico_sub4_2562_withEro3mask_flatten_path.txt')
    
    model.lossFunctionSetting(SHCloss=nn.MSELoss(reduction='sum'))
    
    model.training(totalEpochs = int(args.totalEpochs),
                   optimizer = optimizer,
                   trainNumber = [0])
    
    model.testing(testNumber=[0])
    
    
    
if __name__ == '__main__':
    main()
    
