import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as k
import matplotlib.pyplot as plt
def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice

def create_model():
    unet_model= tf.keras.models.load_model('unet/UNET_Dice_loss_150_epochs.h5',custom_objects={'dice_coef':dice_coef})
    print("UNET MODEL LOADED SUCESSFULLY")
    unet_model.trainable=False
    Xception=tf.keras.applications.Xception(
        input_tensor=tf.keras.layers.InputLayer(input_shape=(32,32,1024)).output,
        weights=None,
        pooling=None,
        classes=2,
        classifier_activation="softmax",
    )
    bottleneck_layer=unet_model.get_layer('conv2d_33').output
    out = Xception(bottleneck_layer)
    classification_model= tf.keras.models.Model(inputs=unet_model.input,outputs=out)
    classification_model.compile(optimizer='sgd',metrics=['accuracy'],loss='binary_crossentropy')
    print("CLASSIFICATION MODEL CREATED SUCESSFULLY")
    return classification_model

def data_generators(PATH, validationSplit=0.1,batchSize=32):
    gen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.,validation_split= validationSplit )
    train_gen= gen.flow_from_directory(PATH, target_size=(512, 512),subset='training',color_mode ="grayscale",batch_size=batchSize)
    val_gen =gen.flow_from_directory(PATH,target_size=(512, 512),subset='validation',color_mode ="grayscale",batch_size=batchSize)
    print(f"GENERATOR OF BATCH SIZE = {batchSize} CREATED SUCCESSFULLY")
    return (train_gen , val_gen)

def train_model(classification_model,train_gen,val_gen,epochs=100,saveplot=True,save_model=True):
    print(f" MODEL TRAINING STARTS FOR {epochs} EPOCHS")
    history=classification_model.fit(train_gen,validation_data=val_gen,epochs=epochs)
    if saveplot== True:
        saveplot(history)
    if save_model== True:
        classification_model.save('Trained_models\classification_model.h5')
        print("MODEL SAVED")

def save_plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train','val'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epochs Vs Loss Graph')
    plt.savefig('Plots\classification_Loss.png')
    print("LOSS PLOT SAVED")

classification_model=create_model()
train_gen,val_gen = data_generators('Dataset')
train_model(classification_model,train_gen,val_gen)

