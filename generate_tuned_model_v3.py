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
from keras.callbacks import ModelCheckpoint
import pandas as pd
from scipy.signal import convolve2d
import custom_loss
import generate_training_patches


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
# Sample: python generate_tuned_model_v3.py --in_geo_path ./binary_raster_md/ --in_tile_path ./binary_raster_md_tif/ --in_model_path_sup ../landcover-old/web_tool/data/naip_demo_model.h5 --in_model_path_ae ../landcover-old/web_tool/data/naip_autoencoder.h5 --out_model_path_sup ./naip_demo_tuned.h5 --out_model_path_ae ./naip_autoencoder_tuned.h5 --num_classes 2 --gpu 1 --exp 1 --even even

def masked_categorical_crossentropy(y_true, y_pred):
    
    mask = K.all(K.equal(y_true, [1,0,0,0,0,0]), axis=-1)
    mask = 1 - K.cast(mask, K.floatx())

    loss = K.categorical_crossentropy(y_true, y_pred) * mask

    return K.sum(loss) / K.sum(mask)

keras.losses.masked_categorical_crossentropy = masked_categorical_crossentropy

def get_loss(mask_value):
    mask_value = K.variable(mask_value)
    def masked_binary_crossentropy(y_true, y_pred):
        
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        loss = K.binary_crossentropy(y_true, y_pred) * mask

        return K.sum(loss) / K.sum(mask)
    return masked_binary_crossentropy

def get_model(model_path, num_classes):
    # K.clear_session()
    tmodel = keras.models.load_model(model_path, custom_objects={
        "jaccard_loss":keras.metrics.mean_squared_error,
        "loss":keras.metrics.mean_squared_error,
        "masked_categorical_crossentropy":keras.metrics.mean_squared_error
    })
    toutput = tmodel.layers[-2].output
    toutput = Conv2D(num_classes+1, (1,1), padding="same", use_bias=True, activation="softmax", name="output_conv")(toutput)
    model = keras.models.Model(inputs=tmodel.inputs, outputs=[toutput])

    optimizer = Adam(lr=0.001)
    loss_mask = np.zeros(num_classes+1)
    loss_mask[0] = 1

    model.compile(loss=K.binary_crossentropy, optimizer=optimizer)
    
    return model

def train_model_from_points(in_geo_path, in_model_path_sup, in_model_path_ae, in_tile_path, out_model_path_sup, out_model_path_ae, num_classes, exp, even):
    # Train supervised
    # print("Loading initial models...")
    # model_sup = get_model(in_model_path_sup, num_classes)
    # model_sup.summary()

    # # Load in sample
    # print("Loading tiles...")
    # x_train, y_train = generate_training_patches.gen_training_patches("../../../media/disk2/datasets/maaryland_naip_2017/",
    #  "./binary_raster_md_tif/", 240, 240, 4, 2, 50000)

    # print(x_train.shape)
    # print(y_train.shape)

    # y_train_ae[:,:,:] = [1] + [0] * (y_train_ae.shape[-1]-1)
    # y_train[:,:,:] = [1] + [0] * (y_train.shape[-1]-1)

    # x_train = x_train / 255.0
    # x_train_ae = x_train_ae / 255.0

    # Supervised tuning

    # print("Tuning supervised model")

    # cpPath = f"./{exp}/tmp_sup_{even}/sup_tuned_model_{even}_"

    # checkpointer_sup = ModelCheckpoint(filepath=(cpPath+"{epoch:02d}_{loss:.2f}.h5"), monitor='loss', verbose=1)

    # model_sup.fit(
    #     x_train, y_train,
    #     batch_size=10, epochs=10, verbose=1, validation_split=0,
    #     callbacks=[checkpointer_sup]
    # )

    # model_sup.save(out_model_path_sup)

    # Unsupervised tuning

    print("Tuning Unsupervised model")

    model_ae = get_model(in_model_path_ae, num_classes)
    model_ae.summary()

    # Load in sample
    print("Loading tiles...")
    x_train_ae, y_train_ae = generate_training_patches.gen_training_patches("../../../media/disk2/datasets/maaryland_naip_2017/",
     "./binary_raster_md_tif/", 150, 150, 4, 2, 25000)

    print(x_train_ae.shape)
    print(y_train_ae.shape)

    cpPath = f"{exp}/tmp_ae_{even}/ae_tuned_model_{even}_"

    checkpointer_ae = ModelCheckpoint(filepath=(cpPath+"{epoch:02d}_{loss:.2f}.h5"), monitor='loss', verbose=1)

    model_ae.fit(
        x_train_ae, y_train_ae,
        batch_size=10, epochs=10, verbose=1, validation_split=0,
        callbacks=[checkpointer_ae]
    )

    model_ae.save(out_model_path_ae)

def main():
    parser = argparse.ArgumentParser(description="Generate a model tuned using webtool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    # Input
    parser.add_argument("--in_geo_path", action="store", dest="in_geo_path", type=str, help="Input geojson dir path (i.e. ../data/output.geojson)", required=True)
    parser.add_argument("--in_tile_path", action="store", dest="in_tile_path", type=str, help="Path to input tif dir", required=True)
    # Models
    parser.add_argument("--in_model_path_sup", action="store", dest="in_model_path_sup", type=str, help="Path to supervised model that needs retraining", required=True)
    parser.add_argument("--in_model_path_ae", action="store", dest="in_model_path_ae", type=str, help="Path to unsupervised model that needs retraining", required=True)
    parser.add_argument("--out_model_path_sup", action="store", dest="out_model_path_sup", type=str, help="Output path for tuned supervised model", required=True)
    parser.add_argument("--out_model_path_ae", action="store", dest="out_model_path_ae", type=str, help="Output path for tuned unsupervised model", required=True)
    # Custom
    parser.add_argument("--num_classes", action="store", dest="num_classes", type=str, help="Number of classes", required=True)
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", required=True)
    # Experiment argument
    parser.add_argument("--exp", action="store",dest="exp", type=str, required=True)
    parser.add_argument("--even", action="store",dest="evenodd", type=str, required=True)

    args = parser.parse_args(sys.argv[1:])
    args.batch_size=10
    args.num_epochs=30

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

    start_time = float(time.time())

    print("GeoJson file read at {}".format(args.in_geo_path))

    print("Retraining model from GeoJson")

    train_model_from_points(args.in_geo_path, args.in_model_path_sup, args.in_model_path_ae, args.in_tile_path, args.out_model_path_sup, args.out_model_path_ae, int(args.num_classes), args.exp, args.evenodd)

    print("Finished in %0.4f seconds" % (time.time() - start_time))
    
    pass

if __name__ == "__main__":
    main()