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

import keras
import keras.backend as K
import keras.callbacks
import keras.utils
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.layers import Input, Dense, Activation, MaxPooling2D, Conv2D, BatchNormalization
from keras.layers import Concatenate, Cropping2D, Lambda
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import pandas as pd
from scipy.signal import convolve2d
import custom_loss

pos_kernels, neg_kernels = custom_loss.get_pos_neg_kernel("../notebooks/Delmarva_PL_House_Final/Delmarva_PL_House_Final.shp")

def masked_categorical_crossentropy(y_true, y_pred):
    
    mask = K.all(K.equal(y_true, [1,0,0,0,0,0]), axis=-1)
    mask = 1 - K.cast(mask, K.floatx())

    loss = K.categorical_crossentropy(y_true, y_pred) * mask

    return K.sum(loss) / K.sum(mask)

keras.losses.masked_categorical_crossentropy = masked_categorical_crossentropy


def get_loss(mask_value):
    mask_value = K.variable(mask_value)
    def custom_loss_fn(y_true, y_pred):   
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        loss = K.categorical_crossentropy(y_true, y_pred) * mask

        temp_p_kernels = (np.zeros((50,50,3,len(pos_kernels))).astype(np.float32))
        temp_n_kernels = (np.zeros((50,50,3,len(neg_kernels))).astype(np.float32))

        # Equal amount of pos and neg kernels
        for i in range(len(pos_kernels)):
            temp_p_kernels[:,:,2,i] = pos_kernels[i]
            print(neg_kernels[i].shape)
            temp_n_kernels[:,:,2,i] = neg_kernels[i]
        
        p_loss = K.mean(K.conv2d(y_pred, (temp_p_kernels), padding='valid', data_format="channels_last"))
        n_loss = K.mean(K.conv2d(y_pred, (temp_n_kernels), padding='valid',data_format="channels_last"))
                
        p_loss = p_loss / (temp_p_kernels.shape[-1])
        n_loss = n_loss / (temp_n_kernels.shape[-1])

        return ((K.sum(loss) / K.sum(mask)) - p_loss + n_loss)
    return custom_loss_fn

def get_model(model_path, num_classes):
    # K.clear_session()
    tmodel = keras.models.load_model(model_path)
    toutput = tmodel.layers[-2].output
    toutput = Conv2D(num_classes+1, (1,1), padding="same", use_bias=True, activation="softmax", name="output_conv")(toutput)
    model = keras.models.Model(inputs=tmodel.inputs, outputs=[toutput])

    optimizer = Adam(lr=0.001)
    loss_mask = np.zeros(num_classes+1)
    loss_mask[0] = 1

    # OR add shape loss as tensor -> model.add_loss(shape_loss)

    model.compile(loss=get_loss(loss_mask), optimizer=optimizer)
    
    return model

def build_train_set(coords, labels, samples, method='uneven', target_cls=4):
    
    unique_cls, count = np.unique(labels, return_counts=True)
    
    if (method == 'even' and ((samples // len(unique_cls)) > min(count))):
        raise Exception('Too many samples requested')
    elif (method == 'uneven' and ((samples // 2) > count[target_cls])
             or ((samples // 2)> (sum(count) - count[target_cls]))):
        raise Exception('Too many samples requested')
    elif samples > coords.shape[0]:
        raise Exception('Too many samples requested')
    
    coords_samples = np.zeros((samples,2), dtype=np.int32)
    labels_samples = np.zeros(samples)
    
    # Sampling without replacement?
    used_idx = set()
    
    # Uneven sampling
    if method == 'uneven':
        each_class_size = samples // 2

        # Random subsampling chicken classes first
        for i in range(each_class_size):
            rand_idx = np.random.randint(coords.shape[0])
            while(labels[rand_idx] != target_cls or (rand_idx in used_idx)):
                rand_idx = np.random.randint(coords.shape[0])
            coords_samples[i] = coords[rand_idx]
            labels_samples[i] = labels[rand_idx]
            used_idx.add(rand_idx) # sampling without replacement?

        # Random subsampling from all non chicken classes
        for i in range(each_class_size):
            rand_idx = np.random.randint(coords.shape[0])
            while (labels[rand_idx] == target_cls or (rand_idx in used_idx)):
                rand_idx = np.random.randint(coords.shape[0])
            coords_samples[i+each_class_size] = coords[rand_idx]
            labels_samples[i+each_class_size] = labels[rand_idx]
            used_idx.add(rand_idx)
            
    # Even sampling
    elif method == 'even':
        # Random subsampling of all classes
        each_class_size = samples // len(unique_cls)
        for c in (unique_cls):
            for i in range(each_class_size):
                rand_idx = np.random.randint(coords.shape[0])
                while(labels[rand_idx] != c or (rand_idx in used_idx)):
                    rand_idx = np.random.randint(coords.shape[0])
                coords_samples[c*each_class_size + i] = coords[rand_idx]
                labels_samples[c*each_class_size + i] = labels[rand_idx]
                used_idx.add(rand_idx) # sampling without replacement?
    else:
        raise Exception('Sampling method not specified')
        
    
    return coords_samples, labels_samples

def train_model_from_points(in_geo_path, in_model_path_sup, in_model_path_ae, in_tile_path, out_model_path_sup, out_model_path_ae, num_classes, exp, even):
    # Train supervised
    print("Loading initial models...")
    model_sup = get_model(in_model_path_sup, num_classes)
    model_sup.summary()

    print("Loading tiles...")
    f = rasterio.open(in_tile_path)
    data = np.rollaxis(f.read(), 0, 3)
    profile = f.profile
    transform = f.profile["transform"]
    src_crs = f.crs.to_string()
    f.close()

    # Extracting labels
    print("Loading new GeoJson file...")
    f = fiona.open(in_geo_path)
    temp_coords = []
    temp_labels = []
    for line in f:
        label = line["properties"]["user_label"]
        geom = fiona.transform.transform_geom(f.crs["init"], src_crs, line["geometry"])
        lon, lat = geom["coordinates"]
        y, x = ~transform * (lon, lat)
        y = int(y)
        x = int(x)
        temp_coords.append((x,y))
        temp_labels.append(label)
    f.close()

    temp_coords = np.array(temp_coords)
    temp_labels = np.array(temp_labels)

    coords, labels = build_train_set(temp_coords, temp_labels, 750, method=even)

    print(coords.shape, labels.shape)
    print(pd.Series(labels).value_counts())

    labels = np.where(labels != 4, 0, 1)

    # x-dim, y-dim, # of bands
    x_train_ae = np.zeros((coords.shape[0], 150, 150, 4), dtype=np.float32)
    x_train = np.zeros((coords.shape[0], 240, 240, 4), dtype=np.float32)

    # x-dim, y-dim, # of classes + dummy index
    y_train_ae = np.zeros((coords.shape[0], 150, 150, num_classes+1), dtype=np.uint8)
    y_train = np.zeros((coords.shape[0], 240, 240, num_classes+1), dtype=np.uint8)

    y_train_ae[:,:,:] = [1] + [0] * (y_train_ae.shape[-1]-1)
    y_train[:,:,:] = [1] + [0] * (y_train.shape[-1]-1)

    for i in range(coords.shape[0]):
        y,x = coords[i]
        label = labels[i]

        # Unsupervised 
        x_train_ae[i] = data[y-75:y+74+1, x-75:x+74+1, :].copy()
        y_train_ae[i,75,75,0] = 0
        y_train_ae[i,75,75,label+1] = 1

        # Supervised
        x_train[i] = data[y-120:y+119+1, x-120:x+119+1, :].copy()
        y_train[i,120,120,0] = 0
        y_train[i,120,120,label+1] = 1
        
    x_train = x_train / 255.0
    x_train_ae = x_train_ae / 255.0

    # Supervised tuning

    print("Tuning supervised model")

    cpPath = f"{exp}/tmp_sup_{even}/sup_tuned_model_{even}_"

    checkpointer_sup = ModelCheckpoint(filepath=(cpPath+"{epoch:02d}_{loss:.2f}.h5"), monitor='loss', verbose=1)

    model_sup.fit(
        x_train, y_train,
        batch_size=10, epochs=10, verbose=1, validation_split=0,
        callbacks=[checkpointer_sup]
    )

    model_sup.save(out_model_path_sup)

    # Unsupervised tuning

    print("Tuning Unsupervised model")

    model_ae = get_model(in_model_path_ae, num_classes)
    model_ae.summary()

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
    parser.add_argument("--in_geo_path", action="store", dest="in_geo_path", type=str, help="Input geojson path (i.e. ../data/output.geojson)", required=True)
    parser.add_argument("--in_model_path_sup", action="store", dest="in_model_path_sup", type=str, help="Path to supervised model that needs retraining", required=True)
    parser.add_argument("--in_model_path_ae", action="store", dest="in_model_path_ae", type=str, help="Path to unsupervised model that needs retraining", required=True)
    parser.add_argument("--in_tile_path", action="store", dest="in_tile_path", type=str, help="Path to input tif file", required=True)
    parser.add_argument("--out_model_path_sup", action="store", dest="out_model_path_sup", type=str, help="Output path for tuned supervised model", required=True)
    parser.add_argument("--out_model_path_ae", action="store", dest="out_model_path_ae", type=str, help="Output path for tuned unsupervised model", required=True)
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