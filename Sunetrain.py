import torch
import torch.nn as nn
import argparse
from SunetBase import ModelBase
from SunetNetBase import Unet_sub3


class mySunetModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super(mySunetModel, self).__init__(*args, **kwargs)
        
    def running(self):
        
        inputsData = self.batchData['indata']
        
        self.batchData['outdata']['fODFonSphSub3'] = self.network(inputsData).squeeze()
        
        self.loss.valueCounting({'fODFonSphloss': (self.batchData['outdata']['fODFonSphSub3'], self.batchData['fODFonSphlabel'].float())})
        
        


def main():
    
    parser = argparse.ArgumentParser(description="device and epoch setting.")
    
    parser.add_argument('--device', '-d', default='cuda:0')
    parser.add_argument('--totalEpochs', '-e', default=200)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    network = Unet_sub3(1,1)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    
    model = mySunetModel(network = network,
                         device = device,
                         batchSize = 480,
                         savePath = "/data/zh/EXP_save/sunet/")
    
    model.dataPathSetting(indata= '/data/zh/path/hcp_dODF_b1000_60_Lmax8_SphVal_Ico_sub3_642_withEro3mask_flatten_path.txt',
                          fODFonSphlabel= '/data/zh/path/hcp_mcsd_FOD_wm_0123_SphVal_Ico_sub3_642_withEro3mask_flatten_path.txt')
    
    model.lossFunctionSetting(fODFonSphloss=nn.MSELoss(reduction='sum'))
    
    model.training(totalEpochs = int(args.totalEpochs),
                   optimizer = optimizer,
                   trainNumber = [0,2,4,6])
    
    model.testing(testNumber=[3,5])
    
    
    
if __name__ == '__main__':
    main()
    
