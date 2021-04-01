import glob
import rasterio
import fiona
import numpy as np
import keras.utils
import os
from PIL import Image
import scipy.misc
from scipy.ndimage import rotate

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_gt_set(train_reg, test_reg, val_reg, gt_set):
    train_set = []
    test_set = []
    val_set = []

    for x in gt_set:
        for y in train_reg:
            if y in x:
                train_set.append(x)
        for y in test_reg:
            if y in x:
                test_set.append(x)
        for y in val_reg:
            if y in x:
                val_set.append(x)

    return train_set, test_set, val_set

def get_patches(ground_truth_set, x_fns, output_data_dir, width, height, total_num_samples, partition_name):
    count = 0
    # Get training set
    while count < total_num_samples:
        # Randomly choose a file from input list
        y_fn = np.random.choice(ground_truth_set)
        folder_name = y_fn.split('/')[6][2:7]
        filename = y_fn.split('/')[6][:26]
        x_fn = x_fns + f"/{folder_name}/" + filename + ".tif"
        
        # import IPython; import sys; IPython.embed(); sys.exit(1)

        # Load input file
        f = rasterio.open(x_fn, "r")
        data = f.read().squeeze()
        data = np.rollaxis(data, 0, 3)
        f.close()

        # Load ground truth file
        f = rasterio.open(y_fn, "r")
        target = f.read().squeeze()
        f.close()

        # Get one-hot-encoding for ground truth
        target_one_hot = keras.utils.to_categorical(target, num_classes=2)
        # Used pixels
        used_x = set()
        
        # Randomly sample patch_size # of 256x256 patch per file
        # TODO fix this
        # TODO vectorize this
        for i in range(640):
            if (count >= total_num_samples):
                break
            x = np.random.randint(width, data.shape[1]-width)
            y = np.random.randint(height, data.shape[0]-height)

            # Reselect until x is new pixel
            while x in used_x:
                x = np.random.randint(width, data.shape[1]-width)
                y = np.random.randint(height, data.shape[0]-height)

            used_x.add(x)

            # Set up x_batch with img data at y,x coords
            img = data[y-(height//2):y+(height//2), x-(width//2):x+(width//2), :].astype(np.float32)

            # FOR DENSE
            mask = target_one_hot[y-(height//2):y+(height//2), x-(width//2):x+(width//2)]

            angle = np.random.randint(0,360)

            img = rotate(img, angle=angle, reshape=False)
            img = img[181-(256//2):181+(256//2),181-(256//2):181+(256//2),:]
            # import pdb; pdb.set_trace(); sys.exit(1)

            mask = rotate(mask, angle=angle, reshape=False)
            mask = mask[181-(256//2):181+(256//2),181-(256//2):181+(256//2)]
            
            # Dump out img, target_one_hot to file
            img_file = os.path.join(output_data_dir, partition_name, 'img', f"img_{count}")
            mask_file = os.path.join(output_data_dir, partition_name, 'mask', f"mask_{count}")
            np.save(img_file, img)
            np.save(mask_file, mask)
            # in __getitem__ index, read file from index and return result
            count+=1

def gen_training_patches_sep(x_fns, y_fns, width, height, channel, target, total_num_samples, region, output_root, train_reg, test_reg, val_reg, pct_train=0.80, pct_validation=0.15, seed=42):
    """
    output_root is directory to write data to (img, masks)
    output_root / region
        / train
            / img / 0 /
            / mask / 0 /
        / validation
            / img
            / mask
        / test
            / img
            / mask
    """

    np.random.seed(seed)

    # TODO use pathlib
    # Output
    output_data_dir = os.path.join(output_root, region)
    for partition_name in ["train", "validation", "test"]:
        os.makedirs(os.path.join(output_data_dir, partition_name, "img"), exist_ok=True)
        os.makedirs(os.path.join(output_data_dir, partition_name, "mask"), exist_ok=True)
    
    num_train = int(pct_train*total_num_samples)
    num_validation = int(pct_validation*total_num_samples)
    num_test = total_num_samples - num_train - num_validation
        
    ground_truth_set = glob.glob(y_fns + "*")

    train_set, test_set, val_set = get_gt_set(train_reg, test_reg, val_reg, ground_truth_set)

    print(num_train, num_test, num_validation) 

    get_patches(train_set, x_fns, output_data_dir, width, height, num_train, "train")
    get_patches(test_set, x_fns, output_data_dir, width, height, num_test, "test")
    get_patches(val_set, x_fns, output_data_dir, width, height, num_validation, "validation")
    
    pass

def main():

    region_dict = {
        "poultry_region_1": "m_38075", 
        "poultry_region_2": "m_38076",
        "poultry_region_3": "m_39075",
        "non_poultry_region_1": "m_39077",
        "non_poultry_region_2": "m_38077"
    }

    ## Rotation Exp 2
    data_root = "../../../mnt/sdc/jason/train/balanced"
    random_data_root = "../../../mnt/sdc/jason/train/random"
    region = "rot_exp2"

    train_reg = [region_dict["poultry_region_1"]]
    test_reg = [region_dict["poultry_region_2"]]
    val_reg = [region_dict["poultry_region_3"]]

    gen_training_patches_sep("../../../data/jason/datasets/md_100cm_2017",
     "../../../data/jason/binary_raster_md_tif/", 362, 362, 4, 2, 100000, region=region, output_root=random_data_root, train_reg=train_reg, test_reg=test_reg, val_reg=val_reg,  pct_train=0.80, pct_validation=0.15, seed=42)

    ## Rotation Exp 3
    region = "rot_exp3"

    train_reg = [region_dict["poultry_region_1"], region_dict["non_poultry_region_1"]]
    test_reg = [region_dict["poultry_region_2"], region_dict["non_poultry_region_2"]]
    val_reg = [region_dict["poultry_region_3"]]

    gen_training_patches_sep("../../../data/jason/datasets/md_100cm_2017",
     "../../../data/jason/binary_raster_md_tif/", 362, 362, 4, 2, 100000, region=region, output_root=random_data_root, train_reg=train_reg, test_reg=test_reg, val_reg=val_reg,  pct_train=0.80, pct_validation=0.15, seed=42)

    pass

if __name__ == "__main__":
    main()
