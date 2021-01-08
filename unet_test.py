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

def do_args(arg_list, name):
    parser = argparse.ArgumentParser(description=name)

    parser.add_argument("--model", action="store", dest="model_fn", type=str, required=True, \
        help="Path to Keras .h5 model file to use"
    )
    
    parser.add_argument("--gpu", action="store", dest="gpu", type=int, required=False, \
        help="GPU id to use",
    )

    return parser.parse_args(arg_list)

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

def get_model():
    # K.clear_session()
    model = sm.Unet(input_shape=(None,None,4), classes = 2, activation='softmax', encoder_weights=None)

    optimizer = Adam(lr=0.0001)

    metrics = [sm.metrics.IOUScore(class_indexes=1), sm.metrics.FScore(beta=1), sm.metrics.Precision(class_indexes=1), sm.metrics.Recall(class_indexes=1)]

    # jaccardLoss = sm.losses.JaccardLoss(class_indexes=1)
    bceLoss = sm.losses.BinaryCELoss()
    # lossFx = jaccardLoss + cceLoss

    model.compile(loss=bceLoss, optimizer=optimizer, metrics=metrics)
    
    return model

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

def test_model(model_fn):
    # UNet tuning

    print("Test Unet model")
    np.random.seed(42)

    # TODO load model

    model = get_model()
    model.load_weights(model_fn)
    model.summary()

    # model = keras.models.load_model(model_fn, custom_objects={
    #     "jaccard_loss":keras.metrics.mean_squared_error,
    #     "loss":keras.metrics.mean_squared_error,
    #     "masked_categorical_crossentropy":keras.metrics.mean_squared_error,
    #     "custom_loss_fn": keras.metrics.mean_squared_error,
    #     "iou_coef": keras.metrics.mean_squared_error,
    #     "iou_score": sm.metrics.IOUScore(),
    #     'f1-score': keras.metrics.mean_squared_error,
    #     'precision': keras.metrics.mean_squared_error,
    #     'recall': keras.metrics.mean_squared_error,
    #     'categorical_crossentropy_plus_jaccard_loss': keras.metrics.mean_squared_error,
    # })
    
    data_root = "../../../data/jason/gen_data/random"
    region = "m_38075"
    
    test_generator = ChickenDataGenerator(
        dataset_dir=os.path.join(data_root, region, "test"),
        batch_size=10,
        shuffle=False
    )

    print("Evaluating")
    score = model.evaluate(test_generator, verbose=3)
    print(score)
    # import IPython; import sys; IPython.embed(); sys.exit(1)
    

def main():
    parser = argparse.ArgumentParser(description="Generate a tuned unet model")
    args = do_args(sys.argv[1:], "Test")
    
    model_fn = args.model_fn


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

    start_time = float(time.time())

    test_model(model_fn=model_fn)

    print("Finished in %0.4f seconds" % (time.time() - start_time))
    
    pass

if __name__ == "__main__":
    main()

