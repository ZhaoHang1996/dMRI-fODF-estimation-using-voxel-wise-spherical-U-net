import nibabel as nib
from scipy import ndimage


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



def main():
    
    for num in objectName:
    
        objname = str(objectName[num])
    
        maskPath = "/data/zh/hcp/" + objname + "/Diffusion/nodif_brain_mask.nii.gz"
        mask = nib.load(maskPath)
    
        affine = mask.affine
        mask = mask.get_fdata()
    
        Erosion_mask = ndimage.binary_erosion(mask, iterations=3)
    
        
        
        savePath = "/data/zh/hcp/" + objname + "/Ero3_mask.nii.gz"
        nib.save(nib.Nifti1Image(Erosion_mask.astype(int), affine), savePath)
    
        print(f"{num} of 15 Done~!")
        
    
    
if __name__ == '__main__':
    main()