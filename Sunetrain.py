import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pyvista
from SunetBase import Database, ModelBase
from SunetNetBase import Unet_10k


class mySunetModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super(mySunetModel, self).__init__(*args, **kwargs)
        
    def running(self, batchData):
        
        self.inputsData = batchData


def main():
    
    # device = torch.device("cpu")
    
    network = Unet_10k(1,1)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    
    criterion = nn.MSELoss()
    
    network.train()
    
    trainData = (loadOdf(),loadOdf())

    trainLoader = DataLoader(Database(*trainData),
                             batch_size = 1)
    
    for i, data in enumerate(trainLoader):
        
        indata, target = data
 
        indata = indata.squeeze().float()
        target = target.squeeze().float()
        indata.requires_grad_()

        optimizer.zero_grad()

        
        prediction = network(indata)
        
        loss = criterion(prediction, target)

        loss.backward()

        optimizer.step()
    
    
if __name__ == '__main__':
    main()
    
