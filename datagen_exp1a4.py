import glob
import rasterio
import fiona
import numpy as np
import keras.utils
import os
from PIL import Image
import scipy.misc
import wandb

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def gen_training_patches(x_fns, y_fns, width, height, channel, target, total_num_samples, region, output_root, all, pct_train=0.80, pct_validation=0.15, seed=42):
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

    # Output
    output_data_dir = os.path.join(output_root, region)
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir, exist_ok=True)

    for partition_name in ["train", "validation", "test"]:
        os.makedirs(os.path.join(output_data_dir, partition_name, "img"), exist_ok=True)
        os.makedirs(os.path.join(output_data_dir, partition_name, "mask"), exist_ok=True)
    
    num_train = int(pct_train*total_num_samples)
    num_validation = int(pct_validation*total_num_samples)
    num_test = total_num_samples - num_train - num_validation
    all_partition_names = [
        *['train' for i in range(num_train)],
        *['validation' for i in range(num_validation)],
        *['test' for i in range(num_test)]
        ]
    np.random.shuffle(all_partition_names)
        
    ground_truth_set = glob.glob(y_fns + "*")

    # import pdb; pdb.set_trace(); sys.exit(1)

    if all==False:
        ground_truth_set = [x for x in ground_truth_set if region in x]

    count = 0
    non_zero_count = 0

    # total_num_samples is # training instances
    while count < total_num_samples:
        # Randomly choose a file from input list
        y_fn = np.random.choice(ground_truth_set)
        folder_name = y_fn.split('/')[2][2:7]
        filename = y_fn.split('/')[2][:26]
        x_fn = x_fns + f"/{folder_name}/" + filename + ".tif"
        # print(x_fn)

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

            # Dump out img, target_one_hot to file
            partition_name = all_partition_names[count]
            img_file = os.path.join(output_data_dir, partition_name, 'img', f"img_{count}")
            mask_file = os.path.join(output_data_dir, partition_name, 'mask', f"mask_{count}")
            np.save(img_file, img)
            np.save(mask_file, mask)
            # in __getitem__ index, read file from index and return result

            # What i get is either [0 1] or [1 0]
            if (len(np.unique(mask, axis=0, return_counts=True)[1]) >= 2):
                non_zero_count += 1
            
            count += 1

            if count % 10000 == 0:
                print("Iteration: {}".format(count))

    with open(f"./random_ratio_{region}.txt", "w+") as f:
        f.write("Ratio of chicken house to non-chicken: {}".format(non_zero_count / count))


# TODO, dump to data directory, not this directory
def gen_training_patches_balanced(x_fns, y_fns, width, height, channel, target, total_num_samples, region, output_root,all, pct_train=0.80, pct_validation=0.15, seed=42):

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

    # Output
    output_data_dir = os.path.join(output_root, region)
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir, exist_ok=True)
    for partition_name in ["train", "validation", "test"]:
        os.makedirs(os.path.join(output_data_dir, partition_name, "img"), exist_ok=True)
        os.makedirs(os.path.join(output_data_dir, partition_name, "mask"), exist_ok=True)
    
    num_train = int(pct_train*total_num_samples)
    num_validation = int(pct_validation*total_num_samples)
    num_test = total_num_samples - num_train - num_validation
    all_partition_names = [
        *['train' for i in range(num_train)],
        *['validation' for i in range(num_validation)],
        *['test' for i in range(num_test)]
        ]
    np.random.shuffle(all_partition_names)
        
    ground_truth_set = glob.glob(y_fns + "*")

    if all == False:
        ground_truth_set = [x for x in ground_truth_set if region in x]

    count = 0
    non_zero_count = 0
    files_wo_chicken_houses = 0

    # import pdb; pdb.set_trace()

    while count < total_num_samples:
        # Randomly choose a file from input list
        y_fn = np.random.choice(ground_truth_set)
        folder_name = y_fn.split('/')[6][2:7]
        filename = y_fn.split('/')[6][:26]
        x_fn = x_fns + f"/{folder_name}/" + filename + ".tif"
        # import pdb; pdb.set_trace(); sys.exit(1)
        print(x_fn)

        # Load input file
        f = rasterio.open(x_fn, "r")
        data = f.read().squeeze()
        data = np.rollaxis(data, 0, 3)
        f.close()
        # print(data.shape)

        # Load ground truth file
        f = rasterio.open(y_fn, "r")
        target = f.read().squeeze()
        f.close()

        # Get one-hot-encoding for ground truth
        target_one_hot = keras.utils.to_categorical(target, num_classes=2)
        # Used pixels
        used_x = set()

        # Poultry pixels
        y_ind,x_ind = np.where(target==1)

        # TODO if too many len(x_ind) == 0, keep loading files

        # Randomly sample patch_size # of 256x256 patch per file
        for i in range(640):
            if (count >= total_num_samples):
                break
            x = np.random.randint(width, data.shape[1]-width)
            y = np.random.randint(height, data.shape[0]-height)

            # Force to choose chicken house pixel for ~50% of data
            # TODO switch to based on random coin flip
            if len(x_ind) != 0 and np.random.randint(10) % 2 == 0:
                rand_index = np.random.randint(0,len(x_ind))
                x = x_ind[rand_index]
                y = y_ind[rand_index]

                temp_count = 0
                # If poultry house pixels out of bounds, random reselect pixel
                while not (x-width >= 0 and x+width < data.shape[1] and y-height >= 0 and y+height < data.shape[0]):
                    rand_index = np.random.randint(0,len(x_ind))
                    x = x_ind[rand_index]
                    y = y_ind[rand_index]
                    temp_count += 1
                    if (temp_count > 5):
                        x = np.random.randint(width, data.shape[1]-width)
                        y = np.random.randint(height, data.shape[0]-height)
                
            # Reselect until x is new pixel
            while x in used_x:
                x = np.random.randint(width, data.shape[1]-width)
                y = np.random.randint(height, data.shape[0]-height)

            used_x.add(x)

            # Set up x_batch with img data at y,x coords
            img = data[y-(height//2):y+(height//2), x-(width//2):x+(width//2), :].astype(np.float32)

            # TODO store img in file for count
            
            # FOR DENSE
            mask = target_one_hot[y-(height//2):y+(height//2), x-(width//2):x+(width//2)]

            partition_name = all_partition_names[count]
            img_file = os.path.join(output_data_dir, partition_name, 'img', f"img_{count}")
            mask_file = os.path.join(output_data_dir, partition_name, 'mask', f"mask_{count}")
            np.save(img_file, img)
            np.save(mask_file, mask)

            # TODO store y_batches in mask file for count

            if (len(np.unique(mask, axis=0, return_counts=True)[1]) >= 2):
                non_zero_count += 1
            
            count += 1

            if count % 10000 == 0:
                print("Iteration: {}".format(count))

    # x_batches /= 255.0
    with open(f"./balanced_ratio_{region}.txt", "w+") as f:
        f.write("Ratio of chicken house to non-chicken: {}".format(non_zero_count / count))
    # print("Ratio of chicken house to non-chicken: {}".format(non_zero_count / count))
    print("# of chicken house patches: {}".format(non_zero_count))


    # return x_batches, y_batches


def main():
    ## Exp 1
    data_root = "../../../mnt/sdc/jason/train/balanced"
    random_data_root = "../../../data/jason/train/random"
    region = "m_38075"
    
    # gen_training_patches("../../../data/jason/datasets/md_100cm_2017",
    #  "./binary_raster_md_tif/", 256, 256, 4, 2, 100000, region=region, output_root=random_data_root, all=False, pct_train=0.80, pct_validation=0.15, seed=42)

    # gen_training_patches_balanced("../../../data/jason/datasets/md_100cm_2017",
    #  "./binary_raster_md_tif/", 256, 256, 4, 2, 100000, region=region, output_root=data_root, all=False, pct_train=0.80, pct_validation=0.15, seed=42)


    ## Exp 4
    region = "exp4"
    # gen_training_patches("../../../data/jason/datasets/md_100cm_2017",
    #  "./binary_raster_md_tif/", 256, 256, 4, 2, 100000, region=region, output_root=random_data_root, all=True, pct_train=0.80, pct_validation=0.15, seed=42)

    gen_training_patches_balanced("../../../data/jason/datasets/md_100cm_2017",
     "../../../data/jason/binary_raster_md_tif/", 256, 256, 4, 2, 100000, region=region, output_root=data_root, all=True, pct_train=0.80, pct_validation=0.15, seed=42)


    # import IPython; import sys; IPython.embed(); sys.exit(1)
    pass

if __name__ == "__main__":
    main()
