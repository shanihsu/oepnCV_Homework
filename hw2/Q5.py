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
    cls_list = ['cat', 'dog']
    path = './petimagedata/test/' +cls_list[random.randint(0,1)] + '/' + str(random.randint(10000, 12499)) + '.jpg' 
    showimg = cv2.imread(path)
    img = image.load_img(path, target_size=(224, 224))
    # 圖像欲處理
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # 對圖像進行分類
    preds = net.predict(x)
    cv2.imshow(cls_list[int(np.argmax(preds, axis=1))],showimg)
    # print ('Predicted:', preds)
    # print(path)
    # print(cls_list[int(np.argmax(preds, axis=1))])

def showaccurancy():
    img = cv2.imread("./accurancy.png", cv2.IMREAD_UNCHANGED)
    cv2.imshow('img',img)

def showscreenshot():
    img = cv2.imread("./screenshot.png")
    cv2.imshow('img',cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_CUBIC))

def showresize():
    left = np.array([1, 2])
    height = np.array([92.00,93.40])
    labels = ['before resize', 'after resize']
    plt.ylim(90, 95)  # 設定y軸範圍
    plt.grid(True) 
    plt.bar(left, height, color='b', width=0.5, tick_label=labels)
    plt.show()

def resizeimage():
    for i in range(0, 500):
        path1 = './petimagedata/train/cat/' + str(i) + '.jpg'
        path2 = './petimagedata/train/dog/' + str(i) + '.jpg' 
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        path1 = './petimagedata/train/cat/' + str(i) + '_1.jpg'
        path1 = './petimagedata/train/dog/' + str(i) + '_1.jpg'
        cv2.imwrite(path1, cv2.resize(img1, (int(0.5*img1.shape[1]), int(0.5*img1.shape[0])), interpolation=cv2.INTER_CUBIC))
        cv2.imwrite(path2, cv2.resize(img2, (int(0.5*img2.shape[1]), int(0.5*img2.shape[0])), interpolation=cv2.INTER_CUBIC))


