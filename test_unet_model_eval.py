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
import glob

def get_metrics(y_pred, gt):
    y_pred = y_pred.astype(int)
    y_pred = y_pred.squeeze()
    
    y_true = np.array(gt)
    y_true = y_true.squeeze()
        
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    assert tp + tn + fp + fn == y_true.size

    n = tp + fp + tn + fn

    acc = (tp+tn)/n

    if (tp + fn == 0 or tp + fn + fp == 0):
        recall = 1
        precision = 1
        iou = 1
    else:
        recall = (tp/(tp+fn))
        precision = tp / (tp + fp)
        iou = tp /(tp + fn + fp)

    # print("IOU: {}".format(iou))
    # print("Accuracy: {}".format(acc))
    # print("Precision: {}".format(precision))
    # print("Recall: {}".format(recall))
    return {"IOU": float(iou), "Accuracy": float(acc), "Precision": float(precision), "Recall": float(recall), "TP": int(tp), "FP": int(fp), "TN": int(tn), "fn": int(fn)}

def get_model():
    # K.clear_session()
    model = sm.Unet(input_shape=(None,None,4), classes = 2, activation='softmax', encoder_weights=None)

    optimizer = Adam(lr=0.0001)

    metrics = [sm.metrics.IOUScore(class_indexes=1), sm.metrics.FScore(beta=1), sm.metrics.Precision(class_indexes=1), sm.metrics.Recall(class_indexes=1)]

    bceLoss = sm.losses.BinaryCELoss()

    model.compile(loss=bceLoss, optimizer=optimizer, metrics=metrics)
    
    return model

def get_all_model_res(test_data_root, region, model_folder, metrics_folder):
    test_generator = ChickenDataGenerator(
        dataset_dir=os.path.join(test_data_root, region, "test"),
        batch_size=64
    )
    test_models = glob.glob(model_folder + "*")
    model = get_model()
    sp = model_folder.split("/")
    output_data_dir = metrics_folder + f'/{sp[5]}/{sp[6]}/'
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir, exist_ok=True)

    test_models = sorted(test_models, reverse=False)

    for tm in test_models:
        # import pdb; pdb.set_trace();
        fn = output_data_dir + f'{tm.split("/")[7].split(".")[0]}' + '_metrics'
        print(fn)
        if (os.path.exists(f'{fn}.json')):
            pass
        else:
            model.load_weights(tm)
            metrics_dict = {}

            # Testing code
            for i, (data,img) in enumerate(test_generator):
                res = model.predict(data)
                for idx in range(res.shape[0]):
                    ground_truth = img[idx]
                    metric = get_metrics(np.argmax(res[idx],axis=2), np.argmax(ground_truth,axis=2))
                    key_name = f"{i}-{idx}"
                    metrics_dict[key_name] = (metric)

            # get metrics
            with open(f'{fn}.json', 'w') as f:
                json.dump(metrics_dict, f,indent=4)

def main():

    test_data_root = "../../../mnt/sdc/jason/train/random"
    test_data_root = "../../../data/jason/train/random"
    region = "exp3"
    # get_all_model_res(test_data_root, region, "../../../mnt/sdc/jason/rot_exp3_exp/random/", "metrics_folder")

    # region = "exp4"
    # get_all_model_res(test_data_root, region, "../../../mnt/sdc/jason/exp8_rotation_exp/random/", "metrics_folder")

    # test_models = glob.glob("../../../data/jason/m_38075_rotation_exp/random/" + "*")
    # print(test_models)
    

    # get_all_model_res(test_data_root, region, "../../../data/jason/exp3_exp/random/", "metrics_folder")
    get_all_model_res(test_data_root, region, "../../../data/jason/exp3_exp/balanced/", "metrics_folder")
    
    # get_all_model_res(test_data_root, region, "../../../data/jason/exp2_exp/random/", "metrics_folder")
    # get_all_model_res(test_data_root, region, "../../../data/jason/exp2_exp/random_24/", "metrics_folder")
    # get_all_model_res(test_data_root, region, "../../../data/jason/exp2_exp/balanced/", "metrics_folder")
    # get_all_model_res(test_data_root, region, "../../../data/jason/exp2_exp/balanced_33/", "metrics_folder")
    # get_all_model_res(test_data_root, region, "../../../mnt/sdc/jason/exp4_exp/balanced/", "metrics_folder")
    
    # test_generator = ChickenDataGenerator(
    #     dataset_dir=os.path.join(test_data_root, region, "test"),
    #     batch_size=64
    # )

    # model = get_model()

    # model.load_weights("../../../../data/jason/m_38075_exp/random/unet_model_35_0.00.h5")

    # metrics_dict = []

    # # Testing code
    # for i, (data,img) in enumerate(test_generator):
    #     res = model.predict(data)
    #     for idx in range(res.shape[0]):
    #         ground_truth = img[idx]
    #         metric = get_metrics(np.argmax(res[idx],axis=2), np.argmax(ground_truth,axis=2))
    #         metrics_dict[i] = (metric)
    
    # with open('test.json', 'wb') as f:
    #     json.dump(metrics_dict, f)

if __name__ == "__main__":
    main()