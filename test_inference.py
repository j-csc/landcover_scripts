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
import tensorflow as tf


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
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import time
import subprocess
import datetime
import argparse

# Library imports
import numpy as np
import pandas as pd

import rasterio

import keras
import keras.backend as K
from keras.losses import categorical_crossentropy
import keras.models
import keras.metrics

def masked_categorical_crossentropy(y_true, y_pred):
    
    mask = K.all(K.equal(y_true, [1,0,0,0,0,0]), axis=-1)
    mask = 1 - K.cast(mask, K.floatx())

    loss = K.categorical_crossentropy(y_true, y_pred) * mask

    return K.sum(loss) / K.sum(mask)

keras.losses.masked_categorical_crossentropy = masked_categorical_crossentropy

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

    parser.add_argument("--input", nargs="+", action="store", dest="input_fn", type=str, required=True, \
        help="Path to filename of file that contains tiles"
    )
    parser.add_argument("--output", action="store", dest="output_base", type=str, required=True, \
        help="Output directory to store predictions"
    )
    parser.add_argument("--model", action="store", dest="model_fn", type=str, required=True, \
        help="Path to Keras .h5 model file to use"
    )
    parser.add_argument("--save_probabilities", action="store_true", default=False, \
        help="Enable outputing grayscale probability maps for each class"
    )
    parser.add_argument("--gpu", action="store", dest="gpu", type=int, required=False, \
        help="GPU id to use",
    )
    parser.add_argument("--superres", action="store_true", dest="superres", default=False, \
        help="Is this a superres model",
    )

    return parser.parse_args(arg_list)

def main():
    program_name = "Model inference script"
    args = do_args(sys.argv[1:], program_name)

    input_fn = args.input_fn
    data_dir = os.path.dirname(input_fn[0])
    output_base = args.output_base
    model_fn = args.model_fn
    save_probabilities = args.save_probabilities
    superres = args.superres

    print("Starting %s at %s" % (program_name, str(datetime.datetime.now())))
    start_time = float(time.time())

    # try:
    #     df = pd.read_csv(input_fn)
    #     fns = df[["naip-new_fn","lc_fn","nlcd_fn"]].values
    # except Exception as e:
    #     print("Could not load the input file")
    #     print(e)
    #     return

    model = keras.models.load_model(model_fn, custom_objects={
        "jaccard_loss":keras.metrics.mean_squared_error,
        "loss":keras.metrics.mean_squared_error
    })

    if superres:
        model = keras.models.Model(input=model.inputs, outputs=[model.outputs[0]])
        model.compile("sgd","mse")
    
    output_shape = model.output_shape[1:]
    input_shape = model.input_shape[1:]
    model_input_size = input_shape[0]
    assert len(model.outputs) == 1, "The loaded model has multiple outputs. You need to specify --superres if this model was trained with the superres loss."

    print(output_shape, input_shape, model_input_size)
    # naip_fn is the tif/mrf file name
    for i in range(len(input_fn)):
        tic = float(time.time())
        curr_fn = os.path.join(input_fn[i])

        print("Running model on %s\t%d/%d" % (curr_fn, i+1, len(input_fn)))

        curr_fid = rasterio.open(curr_fn, "r")
        curr_profile = curr_fid.meta.copy()
        curr_tile = curr_fid.read().astype(np.float32) / 255.0
        curr_tile = np.rollaxis(curr_tile, 0, 3)
        curr_fid.close()

        print(curr_tile.shape, model_input_size, output_shape)

        output = run_model_on_tile(model, curr_tile, model_input_size, output_shape[2], 16)
        # print("Last 5.......")
        # print(output[:,:,1:])

        # print("First 5.......")
        # print(output[:,:,:5])
        output = output[:,:,:4]

        #----------------------------------------------------------------
        # Write out each softmax prediction to a separate file
        #----------------------------------------------------------------
        if save_probabilities:
            output_fn = os.path.basename(curr_fn)[:-4] + "_prob.tif"
            current_profile = curr_profile.copy()
            current_profile['driver'] = 'GTiff'
            current_profile['dtype'] = 'uint8'
            current_profile['count'] = 5
            current_profile['compress'] = "lzw"

            # quantize the probabilities
            bins = np.arange(256)
            bins = bins / 255.0
            output = np.digitize(output, bins=bins, right=True).astype(np.uint8)

            f = rasterio.open(os.path.join(output_base, output_fn), 'w', **current_profile)
            f.write(output[:,:,0], 1)
            f.write(output[:,:,1], 2)
            f.write(output[:,:,2], 3)
            f.write(output[:,:,3], 4)
            f.write(output[:,:,4], 5)
            f.close()

        #----------------------------------------------------------------
        # Write out the class predictions
        #----------------------------------------------------------------
        print("Writing out class predictions.")
        output_classes = np.argmax(output, axis=2).astype(np.uint8)
        output_class_fn = os.path.basename(curr_fn)[:-4] + "_class.tif"

        current_profile = curr_profile.copy()
        current_profile['driver'] = 'GTiff'
        current_profile['dtype'] = 'uint8'
        current_profile['count'] = 1
        current_profile['compress'] = "lzw"
        f = rasterio.open(os.path.join(output_base, output_class_fn), 'w', **current_profile)
        f.write(output_classes, 1)
        f.close()

        print("Finished iteration in %0.4f seconds" % (time.time() - tic))

    print("Finished %s in %0.4f seconds" % (program_name, time.time() - start_time))

if __name__ == "__main__":
    main()