from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge, Input
from keras.applications.resnet50 import ResNet50
from keras.utils import np_utils
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image

import random
import os
import cv2

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def train():
    # 透過 data augmentation 產生訓練與驗證用的影像資料
    NUM_EPOCHS = 5
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 10
    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        "./petimagedata/train",  # this is the target directory
        target_size=IMAGE_SIZE,  # all images will be resized to 224x224
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    validation_datagen = ImageDataGenerator()
    validation_generator = validation_datagen.flow_from_directory(
        './petimagedata/valid',  # this is the target directory
        target_size=IMAGE_SIZE,  # all images will be resized to 224x224
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    # 以訓練好的 ResNet50 為基礎來建立模型，
    # 捨棄 ResNet50 頂層的 fully connected layers
    net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
    x = net.output
    x = Flatten()(x)

    # 增加 DropOut layer
    x = Dropout(0.5)(x)

    # 增加 Dense layer，以 softmax 產生個類別的機率值
    output_layer = Dense(2, activation='softmax', name='softmax')(x)

    # 設定凍結與要進行訓練的網路層
    net_final = Model(inputs=net.input, outputs=output_layer)
    for layer in net_final.layers[:2]:
        layer.trainable = False
    for layer in net_final.layers[2:]:
        layer.trainable = True

    # 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
    net_final.compile(optimizer=Adam(lr=1e-5),
                    loss='categorical_crossentropy', metrics=['accuracy'])

    # 輸出整個網路結構
    print(net_final.summary())

    # 訓練模型
    net_final.fit_generator(train_generator,
                            steps_per_epoch = train_generator.samples // BATCH_SIZE,
                            validation_data = validation_generator,
                            validation_steps = validation_generator.samples // BATCH_SIZE,
                            epochs = NUM_EPOCHS)

    # 儲存訓練好的模型
    net_final.save("resnet50.h5")    

def prediction():
    net = load_model('resnet50.h5')
    cls_list = ['cats', 'dogs']
    img = image.load_img("./petimagedata/test/cat/10000.jpg", target_size=(224, 224))
    # 圖像欲處理
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # 對圖像進行分類
    preds = net.predict(x)
    print ('Predicted:', preds)
    print(np.argmax(preds, axis=1))

train()