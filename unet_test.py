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

from generate_training_patches import ChickenDataGenerator

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
from keras.models import load_model
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

def test_model():
    # UNet tuning

    print("Test Unet model")
    np.random.seed(42)

    # TODO load model
    model = load_model('./unet_model_random.h5', custom_objects={"iou_coef": iou_coef})
    
    data_root = "../../../data/jason/gen/random"
    region = "m_38075"
    
    test_generator = ChickenDataGenerator(
        dataset_dir=os.path.join(data_root, region, "test"),
        batch_size=10,
        shuffle=False,
        # steps_per_epoch=10, # currently total number of examples is steps*batch_size=100, maybe eliminate this in chicken test generator
    )
    
    # all_preds = []
    # all_labels = []
    # for x, y in test_generator:
    #     all_preds.append(model(x)[:,:,1] > 0.5)
    #     all_labels.append(y[:,:,1])
    
    # # model.evaluate(all_labels, all_preds, batch_size=10)    
    # y_true = np.concat(all_labels)
    # y_pred = np.concat(all_preds)
    # from sklearn import metrics
    # recall = metrics.recall_score(y_true, y_pred)
    # precision = metrics.precision_score(y_true, y_pred)
    # print("recall:", recall, "precision", precision)
        
    # res = model.evaluate(test_generator, verbose=2, return_dict=True)
    # print(res)
    y_pred = model.predict(test_generator, verbose=2)
    import IPython; import sys; IPython.embed(); sys.exit(1)
    # print(score)

def main():
    parser = argparse.ArgumentParser(description="Generate a tuned unet model")
    args = parser.parse_args(sys.argv[1:])
    # parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", required=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

    start_time = float(time.time())

    test_model()

    print("Finished in %0.4f seconds" % (time.time() - start_time))
    
    pass

if __name__ == "__main__":
    main()

# python3 generate_unet_model.py --num_classes 2 --gpu 1 --exp unet_test