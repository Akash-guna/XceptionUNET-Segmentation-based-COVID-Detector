# XceptionUNET Segmentation based COVID Detector
 An ensemble of UNET and Xception for COVID detecion using segmented lesions from Lung CT Scans.
 
 Covid was detected by segmenting the anomalous regions from the CT scans of Lungs. The anomalous regions were then classified into COVID and Non Covid Regions. Dice Loss Was used to train the UNET and Binary Crossentropy is used for training the Classifier.
 
 ## Architecture

 <p align="center">
 <img src = "](https://user-images.githubusercontent.com/54630055/153004254-a9911ef5-f010-4e6f-bd50-30f0bd1e9f62.png" width = 600 height=400 >
 </p>
 
 ## Dataset Links:
 
 1. <a href="https://radiopaedia.org/articles/covid-19-4?lang=us">Radiopedia Dataset</a>
 1. <a href="https://radiopaedia.org/articles/covid-19-4?lang=us">Medseg Dataset</a>
 
 ## Trained Models:
 All Trained Models Required Could Be Found Here : <a href="https://drive.google.com/drive/folders/12tIPjW5lPrTx_6ynPu4JXCsOiBQTFZZ0?usp=sharing"> LINK</a>
  
 UNET_Dice_Loss_150_epochs.h5 ----> Trained UNET
 
 Xception_ori.h5 -----> Trained Classification Model


 ### Segmentation Results:
<p align="center">
<img src = "https://user-images.githubusercontent.com/54630055/153003127-dd2d2123-87a2-4fdc-9e20-bc0c7c0438d1.png" width = 600 height=400 >
 </p>
