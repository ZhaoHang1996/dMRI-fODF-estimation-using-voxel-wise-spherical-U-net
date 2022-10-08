import pyvista
import numpy as np
import os

Icos = ['Ico_sub0_12',
        'Ico_sub1_42',
        'Ico_sub2_162',
        'Ico_sub3_642',
        'Ico_sub4_2562',
        'Ico_sub5_10242',
        'Ico_sub6_40962']
'''
We don't need so much sampling points to reconstruct a ODF which has more simple
space structure than a brain cortical surface. Then Ico_sub5 and Ico_sub6 are not
get used.
'''





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


rootDir = '/data/zh/hcp_ODF_SphericalValues_Icosahedron/'


taskQueue = [['/data/zh/hcp_dODF_SH_MRtrix3/b1000_30_Lmax6/', '_dODFSH_b1000_30_Lmax6.nii.gz', 'dODF_MRtrix3/b1000_30_Lmax6/', '_dODF_b1000_30_Lmax6_SphVal_'],
             ['/data/zh/hcp_dODF_SH_MRtrix3/b1000_60_Lmax8/', '_dODFSH_b1000_60_Lmax8.nii.gz', 'dODF_MRtrix3/b1000_60_Lmax8/', '_dODF_b1000_60_Lmax8_SphVal_'],
             ['/data/zh/hcp_dODF_SH_MRtrix3/b1000_90_Lmax8/', '_dODFSH_b1000_90_Lmax8.nii.gz', 'dODF_MRtrix3/b1000_90_Lmax8/', '_dODF_b1000_90_Lmax8_SphVal_'],
             ['/data/zh/hcp_mcsdResult_0123_MRtrix3/FOD/', '_mcsd_FOD_wm_0123.nii.gz', 'fODF_mcsd_0123_MRtrix3/', '_mcsd_FOD_wm_0123_SphVal_'],
             ]



def main():
    
    for i in Icos:
        
        ico = pyvista.read((rootDir+i+".vtk"))
        
        vertices = ico.points
        
        vertPath = rootDir + i + '_vertices.txt'
        
        np.savetxt(vertPath, vertices, fmt='%f', delimiter=' ')
        
        for num in objectName:
            
            objname = str(objectName[num])
    
            for task in taskQueue:
                
                dataPath = task[0] + objname + task[1]
                
                outPath = rootDir + task[2] + objname + task[3] + i + ".nii.gz"
                
                com = f"sh2amp {dataPath} {vertPath} {outPath}"
                print(com)
                os.system(com)
    




if __name__ == '__main__':
    main()








