#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: disable=E1137,E1136,E0110

# export SM_FRAMEWORK=tf.keras



import sys, os, time
import joblib
import argparse
import numpy as np
import glob
import json
import rasterio
import fiona
import fiona.transform
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# Here we look through the args to find which GPU we should use
# We must do this before importing keras, which is super hacky
# See: https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
# TODO: This _really_ should be part of the normal argparse code.
def parse_args(args, key):
    def is_int(s):
        try: 
            int(s)
            return True
        except ValueError:
            return False
    for i, arg in enumerate(args):
        if arg == key:
            if i+1 < len(sys.argv):
                if is_int(args[i+1]):
                    return args[i+1]
    return None
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_ID = parse_args(sys.argv, "--gpu")
if GPU_ID is not None: # if we passed `--gpu INT`, then set the flag, else don't
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import keras
import tensorflow.keras.backend as K
import keras.callbacks
import keras.utils
import tensorflow as tf
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.layers import Input, Dense, Activation, MaxPooling2D, Conv2D, BatchNormalization
from keras.layers import Concatenate, Cropping2D, Lambda
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import segmentation_models as sm
import generate_training_patches_segmentation
import generate_training_patches

# Sample: python generate_tuned_model_v3.py --in_model_path_ae ./naip_autoencoder.h5  --out_model_path_ae ./naip_autoencoder_tuned.h5 --num_classes 2 --gpu 1 --exp test_run --exp_type single_tile_4000s

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def masked_categorical_crossentropy(y_true, y_pred):
    
    mask = K.all(K.equal(y_true, [1,0,0,0,0,0]), axis=-1)
    mask = 1 - K.cast(mask, K.floatx())

    loss = K.categorical_crossentropy(y_true, y_pred) * mask

    return K.sum(loss) / K.sum(mask)

keras.losses.masked_categorical_crossentropy = masked_categorical_crossentropy

def get_loss(mask_value):
    mask_value = K.variable(mask_value)
    def masked_categorical_crossentropy(y_true, y_pred):
        
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        loss = K.categorical_crossentropy(y_true, y_pred) * mask

        return K.sum(loss) / K.sum(mask)
    return masked_categorical_crossentropy

def get_model(num_classes):
    # K.clear_session()
    model = sm.Unet('resnet34', classes = num_classes, activation='softmax')

    optimizer = Adam(lr=0.0001)

    model.compile(loss=K.categorical_crossentropy, optimizer=optimizer, metrics=[iou_coef, tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    
    return model

def train_model_from_points(num_classes, exp):
    # UNet tuning

    print("Tuning Unet model")

    model = get_model(num_classes)
    model.summary()

    # Load in sample
    print("Loading tiles...")

    train_ratio = 0.80
    validation_ratio = 0.15
    test_ratio = 0.05

    dataX, dataY = generate_training_patches.gen_training_patches("../../../data/jason/datasets/md_100cm_2017/38075/",
    "./binary_raster_md_tif/", 256, 256, 4, 2, 100000, region="m_38075")

    # Train is now 80% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio)

    # test is now 5% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

    cpPath = "{}/unet_model_".format(exp)

    checkpointer_ae = ModelCheckpoint(filepath=(cpPath+"{epoch:02d}_{loss:.2f}.h5"), monitor='loss', verbose=1)

    model.fit(
        x_train, y_train,
        batch_size=10, epochs=100, verbose=1,
        validation_data=(x_val, y_val),
        callbacks=[checkpointer_ae]
    )
    model.save("./unet_model.h5")

    print("Testing")
    score=model.evaluate(iou_coef, y_test, verbose=2)
    print(score)

def main():
    parser = argparse.ArgumentParser(description="Generate a tuned unet model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    # Custom
    parser.add_argument("--num_classes", action="store", dest="num_classes", type=str, help="Number of classes", required=True)
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", required=True)
    # Experiment argument
    parser.add_argument("--exp", action="store",dest="exp", type=str, required=True)

    args = parser.parse_args(sys.argv[1:])
    args.batch_size=10
    args.num_epochs=30

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

    start_time = float(time.time())

    train_model_from_points(int(args.num_classes), args.exp)

    print("Finished in %0.4f seconds" % (time.time() - start_time))
    
    pass

if __name__ == "__main__":
    main()

# python3 generate_unet_model.py --num_classes 2 --gpu 1 --exp unet_test