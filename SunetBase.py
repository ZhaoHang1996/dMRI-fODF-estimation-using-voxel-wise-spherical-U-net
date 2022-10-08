import os
import time
import scipy.io as sio 
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter


# neigh_id_path = r"C:\Users\31758\Desktop\SphericalUNetPackage-main\sphericalunet\utils\neigh_indices\\"
neigh_id_path = "/data/zh/EXP_save/sunet/neigh_indices/"


objectName = {
0:599671,
1:601127,
2:613538,
3:620434,
4:622236,
5:623844,
6:627549,
7:638049,
8:644044,
9:645551,
10:654754,
11:665254,
12:672756,
13:673455,
14:677968}

class ModelBase(object):
    def __init__(self, network, device, batchSize, savePath, task=None):
        
        print("\n=================== Start Running ===================\n")
        self.network = network
        self.device = device
        self.batchSize = batchSize
        
        if savePath[-1] != "/":
            savePath += "/"
        self.savePath = savePath
        
        assert (task is not None), "Model Prediction Task Is Not Setting Yet~! "
        self.task = task
        
        
        with open("/data/zh/path/Ero3maskpath.txt") as f:
            maskPath = f.readlines()
        self.maskPath = [x.strip('\r\n') for x in maskPath]
        
        self.network.to(self.device)
        
        print(f"Using device: {self.device}")


    def dataPathSetting(self, **pathFile):
        
        self.dataManager = DataManager(**pathFile)

    def lossFunctionSetting(self, **lossfunctions):
        
        print(f"Loss Functions Used are:\n {lossfunctions}")
        
        self.loss = LossFunctionsManager(**lossfunctions)
        self.loss.writer = SummaryWriter(self.savePath)

    def saveInit(self):
        
        self.outputSaveBag = {}
        
        if 'SHC' in self.task: self.outputSaveBag['SHC'] = np.zeros(self.testDataLength, 45)
        
        if 'PKS' in self.task: self.outputSaveBag['PKS'] = np.zeros(self.testDataLength, 3, 3)
        
        
    def savePred(self):
        
        startPoint = self.saveCounter
        
        if 'SHC' in self.task:
            self.outputSaveBag['SHC'][startPoint:startPoint+self.inputsData.shape[0]] = self.outSHC.cpu().detach().numpy()
            
        if 'PKS' in self.task:
            self.outputSaveBag['PKS'][startPoint:startPoint+self.inputsData.shape[0], 0, :] = self.outPKS1.cpu().detach().numpy()
            self.outputSaveBag['PKS'][startPoint:startPoint+self.inputsData.shape[0], 1, :] = self.outPKS2.cpu().detach().numpy()
            self.outputSaveBag['PKS'][startPoint:startPoint+self.inputsData.shape[0], 2, :] = self.outPKS3.cpu().detach().numpy()
            
        self.saveCounter += self.inputsData.shape[0]
    
    def training(self, totalEpochs, optimizer, trainNumber, saveTempPara=False, paraLoadPath=None):
        
        print("\n=================== Start Training ===================\n")
        print(f"We use object {trainNumber} to train.")
    
        if paraLoadPath is not None:
            self.network.load_state_dict(torch.load(paraLoadPath))
            print(f"Get the preTrained parameters: {paraLoadPath}")
        
        self.latestParaPath = "None"
        
        self.network.train()
        
        self.dataManager.loadData(trainNumber)
        
        for epoch in range(totalEpochs+1):
            
            self.loss.valueInit()
            
            start_time = time.time()
            
            for x in range(len(trainNumber)):
                
                trainData = self.dataManager.readData_order(x)
                
                trainLoader = DataLoader(Database(*trainData),
                                            batch_size = self.batchSize,
                                            shuffle = True)
                
                for i,data in enumerate(trainLoader):
                    
                    batchData = self.dataManager.readLoader(data)
                    
                    for name in batchData: 
                        batchData[name] = batchData[name].to(self.device)
            
                    batchData['indata'] = batchData['indata'].float()
                    
                    batchData['indata'].requires_grad_()
                    
                    optimizer.zero_grad()
                    
                    self.running(batchData)
                    
                    self.loss.lossBackward()
                    
                    optimizer.step()

            cost_time = time.time() - start_time
            
            self.loss.valueAveraged(len(trainNumber))

            self.loss.writeToTensorboard(epoch)

            if not saveTempPara:
                if os.path.exists(self.latestParaPath):
                    os.remove(self.latestParaPath)

            self.latestParaPath = self.savePath + "parameterEpoch" + str(epoch) + ".pkl"
            torch.save(self.network.state_dict(), self.latestParaPath)

            print("Epoch %d cost %.2f s" % epoch, cost_time)
            self.loss.valueShow()
            print("\n")
            
    def testing(self, testNumber, paraLoadPath=None):
        
        print("\n=================== Start Testing ===================\n")
        print(f"We use object {testNumber} to test.")
        
        if paraLoadPath is None:
            paraLoadPath = self.latestParaPath
            
        self.network.load_state_dict(torch.load(paraLoadPath))
        
        self.network.to(self.device)
        self.network.eval()
        
        self.dataManager.loadData(testNumber)
        
        for x, obj in enumerate(testNumber):
            
            print(f"\n========= {objectName[obj]} in Testing =========\n")
        
            testData = self.dataManager.readData_order(x)
        
            testLoader = DataLoader(Database(*testData),
                                    batch_size=self.batchSize,
                                    shuffle = False)
            
            self.testDataLength = len(testLoader.dataset)
            
            mask = nib.load(self.maskPath[obj])
            affine = mask.affine
            self.mask = mask.get_fdata()
            
            self.saveCounter = 0
            self.saveInit()
            
            
            
            self.loss.valueInit()
            
            start_time = time.time()
            
            for i, data in enumerate(testLoader):
                
                batchData = self.dataManager.readLoader(data)
                for name in batchData: 
                    batchData[name] = batchData[name].to(self.device)
                
                batchData['indata'] = batchData['indata'].float()
                
                self.running(batchData)
                
                self.savePred()
                
            assert (self.saveCounter == self.testDataLength)
            
            self.loss.valueShow()
            cost_time = time.time() - start_time
            
            print("Test time cost %.2f s" % cost_time)
            
            self.postManager = DataPostProcessing(self.outputSaveBag, self.mask, obj, self.savePath, affine)
            self.postManager.stretchBack()
            self.postManager.showEval()
            self.postManager.save()
                
    def running(self):
        raise Exception("The running process needs to be done in sub-class~!")
        

class LossFunctionsManager(object):
    
    def __init__(self, **lossfunctions):
        
        self.functions = lossfunctions
        self._writer = None
    
    @property
    def writer(self): return self._writer
    @writer.setter
    def writer(self, wtr): self._writer = wtr
    
    
    def valueInit(self):
        
        self.value = {}
        
        for lossname in self.functions:
            self.value[lossname] = 0.0

    def valueAveraged(self, denominator):
        
        for lossname in self.value:
            self.value /= denominator
            
    def valueCounting(self, lossParas):
        
        assert len(lossParas) == len(self.functions)
        
        self.lossResult = []
        
        for lossname in self.functions:
            
            result = self.functions[lossname](*lossParas[lossname])
            
            self.lossResult.append(result)
            
            self.value[lossname] += result.sum().item()
            
    def valueShow(self):
        for lossname in self.value:
            print("{}: {:3f}".format(lossname, self.value[lossname]), end="  ")
            
    def lossBackward(self):
        
        loss = 0
        
        for x in self.lossResult:
            loss += x
            
        loss.backward()
            
    def writeToTensorboard(self, epoch):
        for lossname in self.value:
            self.writer.add_scalar(lossname, self.value[lossname], global_step=epoch)
        
class DataManager(object):
    def __init__(self, **pathFile):
        self.pathBag = {}        

        for key in pathFile:
            
            with open(pathFile[key]) as f:
                path = f.readlines()
            
            self.pathBag[key] = [x.strip('\r\n') for x in path]

    def loadData(self, loadNumber):
        
        self.dataBag = {}
        
        for dataname in self.pathBag:
            self.dataBag[dataname] = []
            
            for num in loadNumber:
                
                data = np.load(self.pathBag[dataname][num])
                
                self.dataBag[dataname].append(data)


    def readData_order(self, order):
        
        return tuple([self.dataBag[dataname][order] for dataname in self.dataBag])

    def readLoader(self, batchData):
        
        assert (len(batchData)==len(self.dataBag))
        
        tempBag = {}
        counter = 0
        for dataname in self.dataBag:
            tempBag[dataname] = batchData[counter]
            counter += 1
            
        return tempBag

class DataPostProcessing(object):
    def __init__(self, vectorBag, mask, obj, savePath, affine):
        
        self.vectorBag = vectorBag
        self.mask = mask 
        self.objname = str(objectName[obj])
        
        self.savePath = savePath
        self.affine = affine
        
    def stretchBack(self):
        
        self.cubeBag = {}
        for dataName in self.vectorBag:
            self.cubeBag[dataName] = np.zeros((self.mask.shape)+(self.vectorBag[dataName].shape[3:]))
        
        counter = 0
        
        for x in range(self.mask.shape[0]):
            for y in range(self.mask.shape[1]):
                for z in range(self.mask.shape[2]):
                    
                    if self.mask[x,y,z]:
                        
                        for dataName in self.vectorBag:
                            
                            self.cubeBag['SHC'][x,y,z] = self.vectorBag['SHC'][counter]
                        
                        counter += 1
                        
    def save(self): 
        
        for dataName in self.cubeBag:
            
            if (self.cubeBag[dataName].shape) > 4:
                self.cubeBag[dataName] = self.cubeBag[dataName].reshape(self.cubeBag[dataName].shape[:3] + (-1,))
            
            sp = self.savePath + self.objname + "_" + str(dataName) + "_pred.nii.gz"
            
            nib.save(nib.Nifti1Image(self.cubeBag[dataName], self.affine), sp)
        
    def showEval(self):
        
        if 'SHC' in self.cubeBag:
            
            SHC_GT = nib.load("/data/zh/hcp_mcsdResult_0123_MRtrix3/FOD/" + self.objname + "_mcsd_FOD_wm_0123.nii.gz").get_fdata()
        
            acc, badPoint = ACC(self.cubeBag['SHC'], SHC_GT, mask=self.mask)

            avg_acc = acc.sum() / (self.mask.sum() - badPoint)

            print("Mean ACC: %.3f" % avg_acc)

            self.cubeBag['ACC'] = acc


        if 'PKS' in self.cubeBag:
            pass
            # /data/zh/hcp_mcsdResult_0123_MRtrix3/peaks/599671_mcsd_peaks_0123_MRtrix3.nii.gz




        


class Database(Dataset):
    def __init__(self, *datas):

        self.dataBag = []
        for data in datas:
            
            self.dataBag.append(torch.from_numpy(data))
            
    def __getitem__(self,index):
        
        itemBag = []
        
        for data in self.dataBag:
            
            itemBag.append(data[index])
        
        return tuple(itemBag)

    def __len__(self):
        return len(self.dataBag[0])


def ndindex(shape):

    ndi = np.nditer(np.zeros(shape), flags=['multi_index'])

    for _ in ndi:
        yield ndi.multi_index


def ACC(SHC1, SHC2, sh_order=8, mask=None):
    
    assert (SHC1.shape == SHC2.shape), "Different Shape in ACC calculation~!"
    
    assert (sh_order % 2 == 0 and sh_order > 0), "sh_order must be 2,4,6,8..."
    
    c_number = ((sh_order+1)*(sh_order+2))//2

    assert (SHC1.shape[-1] == c_number), f"Need {c_number} coefficients but got {SHC1.shape[-1]}"

    
    if mask is None:
        
        mask = np.ones(SHC1.shape[:-1])
    
    
    acc = np.zeros(mask.shape)
    badPoint = 0
    
    for idx in ndindex(mask.shape):
        
        if mask[idx]:
            
            shc1, shc2 = SHC1[idx], SHC2[idx]
            
            
            deno1 = np.sqrt(np.einsum('i,i->', shc1[1:], shc1[1:]))
            deno2 = np.sqrt(np.einsum('i,i->', shc2[1:], shc2[1:]))
            
            deno = deno1 * deno2
            
            if not deno:
                
                badPoint += 1
            
            else:
                
                numerator = np.einsum('i,i->', shc1[1:], shc2[1:])
                
                acc[idx] = (numerator / deno)
            
    return acc, badPoint
        

def GetNeighIdPath(rotated=0, sub=0):

    neigh_id_bag = {7: neigh_id_path +'adj_mat_order_163842_rotated_' + str(rotated) + '.mat',
                    6: neigh_id_path +'adj_mat_order_40962_rotated_' + str(rotated) + '.mat',
                    5: neigh_id_path +'adj_mat_order_10242_rotated_' + str(rotated) + '.mat',
                    4: neigh_id_path +'adj_mat_order_2562_rotated_' + str(rotated) + '.mat',
                    3: neigh_id_path +'adj_mat_order_642_rotated_' + str(rotated) + '.mat',
                    2: neigh_id_path +'adj_mat_order_162_rotated_' + str(rotated) + '.mat',
                    1: neigh_id_path +'adj_mat_order_42_rotated_' + str(rotated) + '.mat',
                    0: neigh_id_path +'adj_mat_order_12_rotated_' + str(rotated) + '.mat'}

    return neigh_id_bag[sub]


def Get_neighs_order(rotated=0, sub=7):
    
    neigh_order_bag = []
    
    for s in range(sub, -1, -1):
    
        neigh_order_bag.append(get_neighs_order(GetNeighIdPath(rotated=rotated, sub=s)))
    
    return tuple(neigh_order_bag)


def get_neighs_order(order_path):
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    neigh_orders = np.zeros((len(adj_mat_order), 7))
    neigh_orders[:,0:6] = adj_mat_order-1
    neigh_orders[:,6] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)
    
    return neigh_orders

def Get_upconv_index(rotated=0, sub=7):
    
    upconv_index_bag = []
    
    for s in range(sub, 0, -1):
        
        top, down = get_upconv_index(GetNeighIdPath(rotated=rotated, sub=s))
        
        upconv_index_bag.append(top)
        upconv_index_bag.append(down) 
    
    return tuple(upconv_index_bag)

def get_upconv_index(order_path):  
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    adj_mat_order = adj_mat_order -1
    nodes = len(adj_mat_order)
    next_nodes = int((len(adj_mat_order)+6)/4)
    upconv_top_index = np.zeros(next_nodes).astype(np.int64) - 1
    for i in range(next_nodes):
        upconv_top_index[i] = i * 7 + 6
    upconv_down_index = np.zeros((nodes-next_nodes) * 2).astype(np.int64) - 1
    for i in range(next_nodes, nodes):
        raw_neigh_order = adj_mat_order[i]
        parent_nodes = raw_neigh_order[raw_neigh_order < next_nodes]
        assert(len(parent_nodes) == 2)
        for j in range(2):
            parent_neigh = adj_mat_order[parent_nodes[j]]
            index = np.where(parent_neigh == i)[0][0]
            upconv_down_index[(i-next_nodes)*2 + j] = parent_nodes[j] * 7 + index
    
    return upconv_top_index, upconv_down_index