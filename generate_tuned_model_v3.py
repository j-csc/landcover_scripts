#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: disable=E1137,E1136,E0110

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

# How to use: 
# python generate_tuned_model_v3.py --in_tile_path ../../../media/disk2/datasets/all_maryalnd_naip/  --in_model_path_ae ../landcover-old/web_tool/data/naip_autoencoder.h5  --out_model_path_ae ./naip_autoencoder_tuned.h5 --num_classes 2 --gpu 1 --exp 1 --even even


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
import keras.backend as K
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
from scipy.signal import convolve2d
import custom_loss
import generate_training_patches_segmentation


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if K.tensorflow_backend._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        K.tensorflow_backend._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in K.tensorflow_backend._LOCAL_DEVICES if 'device:gpu' in x.lower()]

K.tensorflow_backend._get_available_gpus = _get_available_gpus
print(K.tensorflow_backend._get_available_gpus())
# Sample: python generate_tuned_model_v3.py --in_tile_path ../../../media/disk2/datasets/all_maryalnd_naip/  --in_model_path_ae ../landcover-old/web_tool/data/naip_autoencoder.h5  --out_model_path_ae ./naip_autoencoder_tuned.h5 --num_classes 2 --gpu 1 --exp test_run --exp_type single_tile_4000s

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

def get_model(model_path, num_classes):
    # K.clear_session()
    tmodel = keras.models.load_model(model_path, custom_objects={
        "jaccard_loss":keras.metrics.mean_squared_error,
        "loss":keras.metrics.mean_squared_error,
        "masked_categorical_crossentropy":keras.metrics.mean_squared_error
    })
    toutput = tmodel.layers[-2].output
    toutput = Conv2D(num_classes, (1,1), padding="same", use_bias=True, activation="softmax", name="output_conv")(toutput)
    model = keras.models.Model(inputs=tmodel.inputs, outputs=[toutput])

    optimizer = Adam(lr=0.001)

    model.compile(loss=K.categorical_crossentropy, optimizer=optimizer, metrics=[iou_coef])
    
    return model

def train_model_from_points(in_model_path_ae, in_tile_path, out_model_path_ae, num_classes, exp, exp_type):
    # Unsupervised tuning

    print("Tuning Unsupervised model")

    model_ae = get_model(in_model_path_ae, num_classes)
    model_ae.summary()

    # Load in sample
    print("Loading tiles...")
    
    # X, Y = generate_training_patches_segmentation.gen_training_patches_center_and_dense("../../../media/disk2/datasets/all_maryalnd_naip/",
    #  "./binary_raster_md_tif/", 150, 150, 4, 2, 4000, test=True)

    # Testing single tile m_3807537_ne
    # X, Y = generate_training_patches_segmentation.gen_training_patches_center_and_dense_single("../../../media/disk2/datasets/all_maryalnd_naip/",
    # "./binary_raster_md_tif/", 150, 150, 4, 2, 4000, test=True)

    # Train test split (Takes 3000 for train [1000 from train for validation], Test 1000)
    # x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=0.75, test_size=0.25, random_state=42)

    X = np.load("./x_dense.npy")
    Y = np.load("./y_dense.npy")
    X_val = np.load("./x_dense_val.npy")
    Y_val = np.load("./y_dense_val.npy")

    cpPath = f"{exp}/{exp_type}/ae_tuned_model_"

    checkpointer_ae = ModelCheckpoint(filepath=(cpPath+"{epoch:02d}_{loss:.2f}.h5"), monitor='loss', verbose=1)

    es = EarlyStopping(monitor='iou_coef', min_delta=0.05, patience=3)

    model_ae.fit(
        X, Y,
        batch_size=10, epochs=10, verbose=1, validation_data=(X_val, Y_val),
        callbacks=[checkpointer_ae, es]
    )

    model_ae.save(out_model_path_ae)

    # print("Testing")

    # res = model_ae.evaluate(x_test, y_test)

    # print(res)

def main():
    parser = argparse.ArgumentParser(description="Generate a model tuned using webtool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    # Input
    parser.add_argument("--in_tile_path", action="store", dest="in_tile_path", type=str, help="Path to input tif dir", required=True)
    # Models
    parser.add_argument("--in_model_path_ae", action="store", dest="in_model_path_ae", type=str, help="Path to unsupervised model that needs retraining", required=True)
    parser.add_argument("--out_model_path_ae", action="store", dest="out_model_path_ae", type=str, help="Output path for tuned unsupervised model", required=True)
    # Custom
    parser.add_argument("--num_classes", action="store", dest="num_classes", type=str, help="Number of classes", required=True)
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", required=True)
    # Experiment argument
    parser.add_argument("--exp", action="store",dest="exp", type=str, required=True)
    parser.add_argument("--exp_type", action="store",dest="exp_type", type=str, required=True)

    args = parser.parse_args(sys.argv[1:])
    args.batch_size=10
    args.num_epochs=30

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

    start_time = float(time.time())

    train_model_from_points(args.in_model_path_ae, args.in_tile_path, args.out_model_path_ae, int(args.num_classes), args.exp, args.exp_type)

    print("Finished in %0.4f seconds" % (time.time() - start_time))
    
    pass

if __name__ == "__main__":
    main()