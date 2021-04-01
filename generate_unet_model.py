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
import pickle

# from generate_training_patches import ChickenDataGenerastor

# import wandb
# from wandb.keras import WandbCallback
# wandb.init(project="chicken", entity="chisanch")


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
# import generate_training_patches_segmentation
import generate_training_patches

# Sample: python generate_tuned_model_v3.py --in_model_path_ae ./naip_autoencoder.h5  --out_model_path_ae ./naip_autoencoder_tuned.h5 --num_classes 2 --gpu 1 --exp test_run --exp_type single_tile_4000s

class ChickenDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        dataset_dir, # e.g. os.path.join(data_root, region, "train")
        batch_size,
        steps_per_epoch = None,
        shuffle=False,
        input_size=256,
        output_size=256,
        num_channels=4,
        num_classes=2,
        data_type="uint16",
        rotation=False
    ):
        """Initialization"""
        
        self.dataset_dir = dataset_dir

        self.batch_size = batch_size

        self.input_size = input_size
        self.output_size = output_size
    
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.img_names = np.array(sorted(list(glob.glob(os.path.join(self.dataset_dir, "img", "*")))))
        self.mask_names = np.array(sorted(list(glob.glob(os.path.join(self.dataset_dir, "mask", "*")))))

        if steps_per_epoch is not None:
            self.steps_per_epoch = steps_per_epoch
        else:
            self.steps_per_epoch = len(self.img_names)//batch_size # ??? probably handle last batch size    

        self.shuffle = shuffle
        self.rotation = rotation

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.steps_per_epoch

    def __getitem__(self, index):
        """Generate one batch of data"""
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        img_names = self.img_names[indices]
        mask_names = self.mask_names[indices]
        x_batch = np.zeros(
            (self.batch_size, self.input_size, self.input_size, self.num_channels),
            dtype=np.float32,
        )
        y_batch = np.zeros(
            (self.batch_size, self.output_size, self.output_size, self.num_classes),
            dtype=np.float32,
        )

        for i, (img_file, mask_file) in enumerate(zip(img_names, mask_names)):
            data = np.load(img_file).squeeze()
            data /= 255.0
            x_batch[i] = data[:,:,:self.num_channels]
            mask_data = np.load(mask_file)
            y_batch[i] = mask_data

        return x_batch, y_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.img_names))
        if self.shuffle:
            np.random.shuffle(self.indices)

def iou_coef(y_true, y_pred, smooth=1):
    # import IPython; import sys; IPython.embed(); sys.exit(1)
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

# Train_type: random or balanced, region: m_38075, exp2 - 4 
def train_model_from_points(train_type, region):
    # UNet tuning

    print("Tuning Unet model")

    model = get_model()
    model.summary()
    # model.load_weights('./m_38075_rotation_exp/bxalanced/unet_model_57_-0.00.h5')

    # Load in sample
    print("Loading tiles...")

    train_ratio = 0.80
    validation_ratio = 0.15
    test_ratio = 0.05

    # gen is 1d binary classification, gen_data is one hot encoded -> [0 1]
    data_root = f"../../../mnt/sdc/jason/train/{train_type}"
    data_random_root = "../../../mnt/sdc/jason/train/random"

    train_generator = ChickenDataGenerator(
        dataset_dir=os.path.join(data_root, region, "train"),
        batch_size=32,
        steps_per_epoch=None,
        shuffle=True
        )
    
    validation_generator = ChickenDataGenerator(
        dataset_dir=os.path.join(data_root, region, "validation"),
        batch_size=32,
        steps_per_epoch=None,
        shuffle=True
        )
    
    # es = EarlyStopping(monitor='iou_score', min_delta=0, patience=2)

    cpPath = f"/mnt/sdc/jason/{region}_exp/{train_type}/"

    if not os.path.exists(cpPath):
        os.makedirs(cpPath, exist_ok=True)

    checkpointer = ModelCheckpoint(filepath=(cpPath+"unet_model_{epoch:02d}_{loss:.2f}.h5"),save_weights_only=True, monitor='loss', verbose=1)

    model.fit(
        train_generator,
        epochs=100, verbose=1,
        validation_data=validation_generator,
        workers=8,
        callbacks=[checkpointer]
        # keras.callbacks.ModelCheckpoint(bestmodelPath, save_weights_only=True, save_best_only=True, mode='min')]
    )

    # with open('./{region}_{train_type}_trainHistoryDict', 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)

    # model.save(f"./unet_model_{region}_{train_type}.h5")

    # print("Testing")
    # score = model.evaluate(test_generator, verbose=2)
    # print(score)

    # with open(f"./score_unet_model_{region}_{train_type}.txt", "w+") as f:
    #     f.write(f"{score}")

def main():
    parser = argparse.ArgumentParser(description="Generate a tuned unet model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    # Custom
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", required=True)
    # Experiment argument
    parser.add_argument("--region", action="store",dest="region", type=str, required=True)
    parser.add_argument("--traintype", action="store",dest="traintype", type=str, required=True)

    args = parser.parse_args(sys.argv[1:])
    args.batch_size=10
    args.num_epochs=30

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

    start_time = float(time.time())

    train_model_from_points(args.traintype, args.region)

    print("Finished in %0.4f seconds" % (time.time() - start_time))
    
    pass

if __name__ == "__main__":
    main()

# python3 generate_unet_model.py --gpu 1 --region m_38075 --traintype random
# python3 generate_unet_model.py --gpu 2 --region m_38075 --traintype balanced
# python3 generate_unet_model.py --gpu 3 --region exp2 --traintype random
# python3 generate_unet_model.py --gpu 0 --region exp2 --traintype balanced

# python3 generate_unet_model.py --gpu 1 --region exp3 --traintype random
# python3 generate_unet_model.py --gpu 2 --region exp3 --traintype balanced


# python3 generate_unet_model.py --gpu 1 --region m_38075_rotation --traintype random
# python3 generate_unet_model.py --gpu 2 --region m_38075_rotation --traintype balanced