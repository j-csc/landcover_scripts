#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: skip-file
#
# Caleb Robinson <calebrob6@gmail.com>
# and
# Le Hou <lehou0312@gmail.com>
"""Script for running a saved model file on a list of tiles.
"""
# Stdlib imports
import sys
import os
import custom_loss


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

import time
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.layers import Input, Dense, Activation, MaxPooling2D, Conv2D, BatchNormalization
from keras.layers import Concatenate, Cropping2D, Lambda
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import segmentation_models as sm


import subprocess
import datetime
import argparse

# Library imports
import numpy as np
import pandas as pd

import rasterio

import keras
import keras.backend as K
import segmentation_models as sm

def run_model_on_tile(model, naip_tile, inpt_size, output_size, batch_size):
    down_weight_padding = 40
    height = naip_tile.shape[0]
    width = naip_tile.shape[1]

    stride_x = inpt_size - down_weight_padding*2
    stride_y = inpt_size - down_weight_padding*2

    output = np.zeros((height, width, output_size), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32) + 0.000000001
    kernel = np.ones((inpt_size, inpt_size), dtype=np.float32) * 0.1
    kernel[10:-10, 10:-10] = 1
    kernel[down_weight_padding:down_weight_padding+stride_y,
           down_weight_padding:down_weight_padding+stride_x] = 5

    batch = []
    batch_indices = []
    batch_count = 0

    for y_index in (list(range(0, height - inpt_size, stride_y)) + [height - inpt_size,]):
        for x_index in (list(range(0, width - inpt_size, stride_x)) + [width - inpt_size,]):
            naip_im = naip_tile[y_index:y_index+inpt_size, x_index:x_index+inpt_size, :]

            batch.append(naip_im)
            batch_indices.append((y_index, x_index))
            batch_count+=1

    model_output = model.predict(np.array(batch), batch_size=batch_size, verbose=0)


    for i, (y, x) in enumerate(batch_indices):
        output[y:y+inpt_size, x:x+inpt_size] += model_output[i] * kernel[..., np.newaxis]
        counts[y:y+inpt_size, x:x+inpt_size] += kernel

    return output / counts[..., np.newaxis]

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

def do_args(arg_list, name):
    parser = argparse.ArgumentParser(description=name)

    parser.add_argument("--input_fns", nargs="+", action="store", type=str, required=True, \
        help="Filenames of input raster files"
    )
    parser.add_argument("--output_fns", nargs="+", action="store", type=str, required=True, \
        help="Filenames to write output (should be one for each of --input_fns)"
    )
    parser.add_argument("--model", action="store", dest="model_fn", type=str, required=True, \
        help="Path to Keras .h5 model file to use"
    )
    parser.add_argument("--save_probabilities", action="store_true", default=False, \
        help="Enable storing the class probability values for each class"
    )
    parser.add_argument("--gpu", action="store", dest="gpu", type=int, required=False, \
        help="GPU id to use",
    )

    return parser.parse_args(arg_list)

def get_model(num_classes):
    # K.clear_session()
    model = sm.Unet(input_shape=(None,None,4), classes = 1, activation='sigmoid', encoder_weights=None)

    optimizer = Adam(lr=0.0001)

    metrics = [sm.metrics.IOUScore(smooth=1e-05), sm.metrics.FScore(beta=1), sm.metrics.Precision(), sm.metrics.Recall()]

    # jaccardLoss = sm.losses.JaccardLoss(class_indexes=1)
    bceLoss = sm.losses.BinaryCELoss()
    # lossFx = jaccardLoss + cceLoss

    model.compile(loss=bceLoss, optimizer=optimizer, metrics=metrics)
    
    return model

def main():
    program_name = "Model inference script"
    args = do_args(sys.argv[1:], program_name)

    input_fns = args.input_fns
    data_dir = os.path.dirname(input_fns[0])
    output_fns = args.output_fns
    model_fn = args.model_fn
    save_probabilities = args.save_probabilities

    print("Starting %s at %s" % (program_name, str(datetime.datetime.now())))
    start_time = float(time.time())

    assert len(input_fns) == len(output_fns), "Must have the same number of input filenames as output filenames"
    for fn in output_fns:
        assert not os.path.exists(fn), "Output would overwrite existing data: %s" % (fn)
    for fn in input_fns:
        assert os.path.exists(fn), "Input does not exist: %s" % (fn)

    model = keras.models.load_model(model_fn, custom_objects={
        "jaccard_loss":keras.metrics.mean_squared_error,
        "loss":keras.metrics.mean_squared_error,
        "masked_categorical_crossentropy":keras.metrics.mean_squared_error,
        "custom_loss_fn": keras.metrics.mean_squared_error,
        "iou_coef": keras.metrics.mean_squared_error,
        "iou_score": sm.metrics.IOUScore(),
        'f1-score': keras.metrics.mean_squared_error,
        'precision': keras.metrics.mean_squared_error,
        'recall': keras.metrics.mean_squared_error,
        'categorical_crossentropy_plus_jaccard_loss': keras.metrics.mean_squared_error,
    })

    # model = get_model(1)
    # model.load_weights(model_fn)

    output_shape = model.output_shape[1:]
    input_shape = model.input_shape[1:]
    model_input_size = input_shape[0]

    assert len(model.outputs) == 1, "The loaded model has multiple outputs."

    print("Expected input shape: %s" % (str(input_shape)))
    print("Model output shape: %s" % (str(output_shape)))
    
    for fn_idx in range(len(input_fns)):
        tic = float(time.time())
        curr_fn = os.path.join(input_fns[fn_idx])

        print("Running model on %s\t%d/%d" % (curr_fn, fn_idx+1, len(input_fns)))

        curr_fid = rasterio.open(curr_fn, "r")
        curr_profile = curr_fid.meta.copy()
        curr_tile = curr_fid.read().astype(np.float32) / 255.0
        curr_tile = np.rollaxis(curr_tile, 0, 3)
        curr_fid.close()

        print(curr_tile.shape)

        output = run_model_on_tile(model, curr_tile, 256, 2, 16)

        #----------------------------------------------------------------
        # Write out the class predictions
        #----------------------------------------------------------------
        print("Writing out class predictions.")
        output_classes = np.argmax(output, axis=2).astype(np.uint8)

        current_profile = curr_profile.copy()
        current_profile['driver'] = 'GTiff'
        current_profile['dtype'] = 'uint8'
        current_profile['count'] = 1
        current_profile['compress'] = "lzw"
        f = rasterio.open(output_fns[fn_idx], 'w', **current_profile)
        f.write(output_classes, 1)
        f.close()

        print("Finished iteration in %0.4f seconds" % (time.time() - tic))

    print("Finished %s in %0.4f seconds" % (program_name, time.time() - start_time))

if __name__ == "__main__":
    main()

# m_3807537_nw_18_1_20170611.tif