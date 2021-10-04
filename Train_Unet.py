from models.unet import unet
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import os
def load_data():
    file = os.path.join('Unet_dataset','images_medseg.npy')
    input_imgs=np.load(file)
    input_imgs/=255.
    file = os.path.join('Unet_dataset','images_radiopedia.npy')
    input_imgs_rad=np.load(file)
    input_imgs_rad/=255.
    file = os.path.join('Unet_dataset','masks_medseg.npy')
    mask_imgs=np.load(file)
    file =os.path.join('Unet_dataset','masks_radiopedia.npy')
    mask_imgs_rad=np.load(file)
    inps=[]
    for i in range(input_imgs.shape[0]):
        inps.append(input_imgs[i])
    for i in range(input_imgs_rad.shape[0]):
        inps.append(input_imgs_rad[i])
    inps=np.array(inps)
    masks=[]
    for i in range(mask_imgs.shape[0]):
        masks.append(mask_imgs[i])
    for i in range(mask_imgs_rad.shape[0]):
        masks.append(mask_imgs_rad[i])
    gnd_masks=np.array([m[:,:,0] for m in masks])
    cnd_masks=np.array([m[:,:,1] for m in masks])
    msk= np.array([np.expand_dims(gnd_masks[i]+cnd_masks[i],axis=2) for i in range(gnd_masks.shape[0])])
    return (inps,msk)

def train_model(inps,msk,epochs=150,batch_size=8):
    def dice_coef(y_true, y_pred, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
        return dice
    model=unet(None,(512,512,1))
    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = dice_coef)
    model.fit(x=np.array(inps),y=msk,epochs=epochs,batch_size=batch_size)
    model.save('Trained_models/unet_model.h5')

inps,msk=load_data()
train_model(inps,msk)