import numpy as np
import nibabel as nib

Icos = ['Ico_sub0_12',
        'Ico_sub1_42',
        'Ico_sub2_162',
        'Ico_sub3_642',
        'Ico_sub4_2562']

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


taskQueue = [['/data/zh/hcp_ODF_SphericalValues_Icosahedron/dODF_MRtrix3/b1000_30_Lmax6/', '_dODF_b1000_30_Lmax6_SphVal_'],
             ['/data/zh/hcp_ODF_SphericalValues_Icosahedron/dODF_MRtrix3/b1000_60_Lmax8/', '_dODF_b1000_60_Lmax8_SphVal_'],
             ['/data/zh/hcp_ODF_SphericalValues_Icosahedron/dODF_MRtrix3/b1000_90_Lmax8/', '_dODF_b1000_90_Lmax8_SphVal_'],
             ['/data/zh/hcp_ODF_SphericalValues_Icosahedron/fODF_mcsd_0123_MRtrix3/', '_mcsd_FOD_wm_0123_SphVal_'],
             ]



def main():

    for i in Icos:

        for num in objectName:

            objname = str(objectName[num])

            maskPath = "/data/zh/hcp/" + objname + "/Ero3_mask.nii.gz"

            mask = nib.load(maskPath).get_fdata()

            voxelTotalNum = int(mask.sum())

            for task in taskQueue:

                dataPath = task[0] + objname + task[1] + i + ".nii.gz"

                flattenPathPath = "/data/zh/path/hcp" + task[1] + i + "_withEro3mask_flatten_path.txt"

                f_flattenData = open(flattenPathPath, mode='a+')

                data = nib.load(dataPath).get_fdata()

                flattenData = np.zeros((voxelTotalNum, data.shape[-1]))


                lineCounter = 0
                for x in range(mask.shape[0]):
                    for y in range(mask.shape[1]):
                        for z in range(mask.shape[2]):

                            if mask[x, y, z]:

                                flattenData[lineCounter, :] = data[x,y,z,:]

                                lineCounter += 1


                assert (lineCounter==voxelTotalNum)

                flattenDataPath = task[0] + objname + task[1] + i + "_withEro3mask_flatten.npy"

                # nib.save(nib.Nifti1Image(flattenData, affine), flattenDataPath)
                np.save(flattenDataPath, flattenData)
                
                f_flattenData.write(flattenDataPath+str("\n"))
                
                print(flattenDataPath,"is DONE~! \n")

                f_flattenData.close()
    

if __name__ == '__main__':
    main()