import keras
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import rotate
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.layers import Input, Dense, Activation, MaxPooling2D, Conv2D, BatchNormalization
from keras.layers import Concatenate, Cropping2D, Lambda
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import segmentation_models as sm
import json
from generate_training_patches import ChickenDataGenerator

def get_metrics(y_pred, gt):
    y_pred = y_pred.astype(int)
    y_pred = y_pred.squeeze()
    
    y_true = np.array(gt)
    y_true = y_true.squeeze()
    
    uniq = np.unique(y_true)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    assert tp + tn + fp + fn == y_true.size

    n = tp + fp + tn + fn

    acc = (tp+tn)/n
    recall = (tp/(tp+fn))
    precision = tp / (tp + fp)
    iou = tp /(tp + fn + fp)

    if (tp == 0):
        recall = 1
        precision = 1
        iou = 1

    print("IOU: {}".format(iou))
    print("Accuracy: {}".format(acc))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    return {"IOU": iou, "Accuracy": acc, "Precision": precision, "Recall": recall}

def get_model():
    # K.clear_session()
    model = sm.Unet(input_shape=(None,None,4), classes = 2, activation='softmax', encoder_weights=None)

    optimizer = Adam(lr=0.0001)

    metrics = [sm.metrics.IOUScore(class_indexes=1), sm.metrics.FScore(beta=1), sm.metrics.Precision(class_indexes=1), sm.metrics.Recall(class_indexes=1)]

    bceLoss = sm.losses.BinaryCELoss()

    model.compile(loss=bceLoss, optimizer=optimizer, metrics=metrics)
    
    return model

def main():
    test_data_root = "../../../../data/jason/test/random"
    region = "m_38075"

    test_generator = ChickenDataGenerator(
        dataset_dir=os.path.join(test_data_root, region, "test"),
        batch_size=64
    )

    model = get_model()

    model.load_weights("../../../../data/jason/m_38075_exp/random/unet_model_35_0.00.h5")

    metrics_dict = []

    # Testing code
    for i, (data,img) in enumerate(test_generator):
        res = model.predict(data)
        for idx in range(res.shape[0]):
            ground_truth = img[idx]
            metric = get_metrics(np.argmax(res[idx],axis=2), np.argmax(ground_truth,axis=2))
            metrics_dict[i] = (metric)
    
    with open('test.json', 'wb') as f:
        json.dump(metrics_dict, f)

if __name__ == "__main__":
    main()