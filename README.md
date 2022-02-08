# XceptionUNET Segmentation based COVID Detector
 An ensemble of UNET and Xception for COVID detecion using segmented lesions from Lung CT Scans.
 
 Covid was detected by segmenting the anomalous regions from the CT scans of Lungs. The anomalous regions were then classified into COVID and Non Covid Regions.
 
 ## Dataset Links:
 
 1. <a href="https://radiopaedia.org/articles/covid-19-4?lang=us">Radiopedia Dataset</a>
 1. <a href="https://radiopaedia.org/articles/covid-19-4?lang=us">Medseg Dataset</a>
 
 ## Trained Models:
 All Trained Models Required Could Be Found Here : <a href="https://drive.google.com/drive/folders/12tIPjW5lPrTx_6ynPu4JXCsOiBQTFZZ0?usp=sharing"> LINK</a>
  
 UNET_Dice_Loss_150_epochs.h5 ----> Trained UNET
 
 Xception_ori.h5 -----> Trained Classification Model


 ### Segmentation Results:

![seg_res](https://user-images.githubusercontent.com/54630055/153003127-dd2d2123-87a2-4fdc-9e20-bc0c7c0438d1.png)
