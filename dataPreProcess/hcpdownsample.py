import nibabel as nib
import numpy as np
from dipy.io.gradients import read_bvals_bvecs

def NanToZero(data):
    """
    把数据data中的异常值(NAN和INF)置零。
    """
    data[np.isinf(data)] = 0
    data[np.isnan(data)] = 0
    return data

def BvalNormalization(bvalarr, threshold=100):
    '''
    b-values are clustered to allow small deviations
    '''
    
    
    normalizied_bvalarr = np.zeros(len(bvalarr), dtype=bvalarr.dtype)

    normalizied_bvalarr[bvalarr < 0] = -1
    if (-1 in normalizied_bvalarr):
        print(normalizied_bvalarr)
        raise Exception("b-value can't be Negative~!")
    
    threshold = int(threshold)

    for x in [0,1000,2000,3000]:

        normalizied_bvalarr[np.logical_and(bvalarr < (x +threshold) , bvalarr >= (x-threshold))] = x
    
    return normalizied_bvalarr

def hcpdownsample(datapath, bvalspath, bvecspath, normalize=False,
                  b0_num='all', b1000_num=0, b2000_num=0, b3000_num=0):
    """
    Parameters
    ----------
    datapath :
        The path of whole DWI data with full shell and directions.
    bvalspath :
        The path of corresponding bvals with the DWI data.
    bvecspath :
        The path of corresponding bvecs with the DWI data.
    
    b0_num : int or 'all', optional
        The number of gradients of b0 image.(default: all).
        
        WARRING: Note that no matter how many b0 data you want, the
        result have only one b0 data that averaged from all b0 data
        you select.
        
        
    b1000_num : int or 'all', optional
        The number of gradients of b1000 image. The default is 0.
    b2000_num : int or 'all', optional
        The number of gradients of b2000 image. The default is 0.
    b3000_num : int or 'all', optional
        The number of gradients of b3000 image. The default is 0.
    normalize : bool, optional
        Whether to compute the average data of all selceted b0 image and
        normalized the selected data with it.(default: False).
   

    Returns
    -------
    downsampledata : list,
    The subsampled result packed with [data, affine, bvals, bvecs].

    """
    
    print(f"Make a downSampled HCP data for you consistent of {b0_num} b0, {b1000_num} b1000, {b2000_num} b2000, {b3000_num} b3000. ")
    
    
    #读取b值和b向量
    bvals, bvecs = read_bvals_bvecs(bvalspath, bvecspath)
    bvals = BvalNormalization(bvals)

    #读取dmri数据data和掩膜mask
    data = nib.load(datapath)
    affine = data.affine
    data = data.get_fdata()

    #筛选b0数据
    sel_b0 = np.where(bvals == 0)[0]
    fullb0num = len(sel_b0)

    if b0_num == 'all':
        b0_num = fullb0num
    elif not (0 <= b0_num <= fullb0num) :
        raise Exception(f"b0 number must be positive and less than {fullb0num}")

    data_b0 = data[..., sel_b0]

    #筛选b1000数据
    sel_b1000 = np.where(bvals == 1000)[0]
    fullb1000num = len(sel_b1000)
    if b1000_num == 'all':
        b1000_num = fullb1000num
    elif not (0 <= b1000_num <= fullb1000num) :
        raise Exception(f"b1000 number must be positive and less than {fullb1000num}")

    data_b1000 = data[..., sel_b1000]
    bvecs_b1000 = bvecs[sel_b1000, :]

    #筛选b2000数据
    sel_b2000 = np.where(bvals == 2000)[0]
    fullb2000num = len(sel_b2000)
    if b2000_num == 'all':
        b2000_num = fullb2000num
    elif not (0 <= b2000_num <= fullb2000num) :
        raise Exception(f"b2000 number must be positive and less than {fullb2000num}")

    data_b2000 = data[..., sel_b2000]
    bvecs_b2000 = bvecs[sel_b2000, :]

    #筛选b3000数据
    sel_b3000 = np.where(bvals == 3000)[0]
    fullb3000num = len(sel_b3000)
    if b3000_num == 'all':
        b3000_num = fullb3000num
    elif not (0 <= b3000_num <= fullb3000num) :
        raise Exception(f"b3000 number must be positive and less than {fullb3000num}")

    data_b3000 = data[..., sel_b3000]
    bvecs_b3000 = bvecs[sel_b3000, :]
    
    
    
    
    total_num = 1 + b1000_num + b2000_num + b3000_num

    if total_num > 0:

        counter = 0
        #准备一个全为零的array来写入data
        subdata = np.zeros(data.shape[:-1] + (total_num,))

        subvals = np.zeros((total_num,))
        subvecs = np.zeros((total_num, 3))
        
        
        #b0
        avgb0 = np.zeros(data_b0.shape[:-1])
        for x in range(b0_num):
            avgb0 += data_b0[:, :, :, x]
        avgb0 = avgb0 / b0_num

        subdata[..., 0] = avgb0
        subvals[0] = 0.0
        subvecs[0, :] = 0.0
            
        counter += 1
        
        #b1000
        for x in range(b1000_num):
            subdata[:, :, :, counter] = data_b1000[:, :, :, x]
            subvals[counter] = 1000.0
            subvecs[counter, :] = bvecs_b1000[x, :]
            counter += 1
        #b2000
        for x in range(b2000_num):
            subdata[:, :, :, counter] = data_b2000[:, :, :, x]
            subvals[counter] = 2000
            subvecs[counter, :] = bvecs_b2000[x, :]
            counter += 1
        #b3000
        for x in range(b3000_num):
            subdata[:, :, :, counter] = data_b3000[:, :, :, x]
            subvals[counter] = 3000
            subvecs[counter, :] = bvecs_b3000[x, :]
            counter += 1

        
        if normalize:
            nmldata = np.zeros_like(subdata)
            b0data = subdata[..., 0]

            for x in range(subdata.shape[-1]):
                nmldata[..., x] = subdata[..., x] / b0data
            subdata = nmldata

        downsamplePack = []

        downsamplePack.append(NanToZero(subdata))
        downsamplePack.append(affine)
        downsamplePack.append(subvals)
        downsamplePack.append(subvecs)

        return downsamplePack

    else:
        
        raise Exception("totalNum in hcpDownSample is ZERO~!")


def main():
    
    import os
    
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
    
    
    for num in objectName:
        
        objname = str(objectName[num])
        
        dataPath = "/data/zh/hcp/" + objname + "/Diffusion/data.nii.gz"
        bvalsPath = "/data/zh/hcp/" + objname + "/Diffusion/bvals"
        bvecsPath = "/data/zh/hcp/" + objname + "/Diffusion/bvecs"
        
        
        locDir = "/data/zh/hcp_downSample/b1000_30/" + objname + "/"
        if not os.path.exists(locDir):
            os.makedirs(locDir)
        
        
        dataSavePath =  locDir + "b1000_30_downdata.nii.gz"
        bvalSavePath = locDir + "b1000_30_downbvals.txt"
        bvecsSavePath = locDir + "b1000_30_downbvecs.txt"
        
        data, affine, bvals, bvecs =  hcpdownsample(dataPath, bvalsPath, bvecsPath, b0_num='all', b1000_num=30)
    
        nib.save(nib.Nifti1Image(data, affine), dataSavePath)
        np.savetxt(bvalSavePath, bvals, fmt='%d', delimiter=' ')
        np.savetxt(bvecsSavePath, bvecs, fmt='%f', delimiter=' ')
    
        #MRtrix3 convert mif
        
        outPath = locDir + "b1000_30_downdata.mif"
        com = str(f"mrconvert -fslgrad {bvecsSavePath} {bvalSavePath} {dataSavePath} {outPath}")
        
        os.system(com)
        print(num+1,"of 15 is DONE~!")
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()




