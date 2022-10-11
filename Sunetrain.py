import torch
import torch.nn as nn
import argparse
import os
import nibabel as nib
from SunetBase import ModelBase, ACC
from SunetNetBase import Unet_sub3


class mySunetModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super(mySunetModel, self).__init__(*args, **kwargs)
        
    def running(self):
        
        inputsData = self.batchData['indata']
        
        self.batchData['outdata']['fODFonSphSub3'] = self.network(inputsData).squeeze()
        
        self.loss.valueCounting({'fODFonSphloss': (self.batchData['outdata']['fODFonSphSub3'], self.batchData['fODFonSphlabel'].float())})
        
    def evaluation(self):
        
        fODF = self.cubeData['fODFonSphSub3']

        fODFpath = self.postManager.save('fODFonSphSub3', fODF)

        SHpath = str(self.savePath + self.postManager.objname + "_fODF_SHC_pred.nii.gz")

        directionPath = "/data/zh/hcp_ODF_SphericalValues_Icosahedron/Ico_sub3_642_vertices.txt"

        com = f"amp2sh -quiet -lmax 8 -directions {directionPath} {fODFpath} {SHpath}"
        
        os.system(com)
      
        SHC_GT_path = "/data/zh/hcp_mcsdResult_0123_MRtrix3/FOD/" + self.postManager.objname + "_mcsd_FOD_wm_0123.nii.gz"
        SHC_GT = nib.load(SHC_GT_path).get_fdata()
        
        
        acc, badPoint = ACC(nib.load(SHpath).get_fdata(), SHC_GT, mask = self.postManager.mask)
        
        avg_acc = acc.sum() / (self.postManager.totalNumber - badPoint)

        print("Mean ACC: %.3f" % avg_acc)
        
        self.postManager.save('ACC', acc)






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
    
