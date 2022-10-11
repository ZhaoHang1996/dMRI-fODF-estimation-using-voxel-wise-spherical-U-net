import os
from SunetBase import ACC, DataPostProcessing
import nibabel as nib

def main():


    dataPath = "620434_fODFonSphSub3_pred.nii.gz"
    SHpath = "620434_fODF_SH_pred.nii.gz"
    
    dirPath = "/data/zh/hcp_ODF_SphericalValues_Icosahedrons/Ico_sub3_642_vertices.txt"
    
    
    com = f"amp2sh -lmax 8 -directions {dirPath} {dataPath} {SHpath}"
    
    os.system(com)
    
    
    mask = nib.load("/data/zh/hcp/620434/Ero3_mask.nii.gz")
    affine = mask.affine
    mask = mask.get_fdata()
    
    SHC_GT = nib.load("/data/zh/hcp_mcsdResult_0123_MRtrix3/FOD/620434_mcsd_FOD_wm_0123.nii.gz").get_fdata()
    
    
    badPoint, acc = ACC(nib.load(SHpath).get_fdata(), SHC_GT)
    
    avg_acc = acc.sum() / (int(mask.sum()) - badPoint)

    print("Mean ACC: %.3f" % avg_acc)
    
    k = DataPostProcessing(None, None, "620434", "/data/zh/EXP_save/sunet/", affine)
    
    k.save('ACC', acc)
    
    
    

if __name__ == "__main__":
    main()



