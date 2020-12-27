import glob
import rasterio
import fiona
import numpy as np
import keras.utils
import os
from PIL import Image
import scipy.misc

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# TODO:
# data loader
# shuffling data loader
# train-test split loader
# check if there is a good dataloader for this (images + labels)

class ChickenDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        dataset_dir, # e.g. os.path.join(data_root, region, "train")s
        batch_size,
        steps_per_epoch = None,
        shuffle=True,
        input_size=256,
        output_size=256,
        num_channels=4,
        num_classes=2,
        data_type="uint16",
    ):
        """Initialization"""
        
        self.dataset_dir = dataset_dir

        self.batch_size = batch_size

        self.input_size = input_size
        self.output_size = output_size
    
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.img_names = np.array(list(glob.glob(os.path.join(self.dataset_dir, "img", "*"))))
        self.mask_names = np.array(list(glob.glob(os.path.join(self.dataset_dir, "mask", "*"))))
        if steps_per_epoch is not None:
            self.steps_per_epoch = steps_per_epoch
        else:
            self.steps_per_epoch = len(self.img_names)//batch_size # ??? probably handle last batch size    

        self.shuffle = shuffle

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
            # mask_data = ml.squeeze()
            y_batch[i] = mask_data
            # data = np.rollaxis(data, 0, 3)
            # print(data.shape)
                      
        return x_batch, y_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.img_names))
        if self.shuffle:
            np.random.shuffle(self.indices)

# TODO, dump to data directory, not this directory

def gen_training_patches(x_fns, y_fns, width, height, channel, target, total_num_samples, region, output_root="./data", pct_train=0.80, pct_validation=0.15, seed=42):
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
    # x_batches = np.zeros((total_num_samples, width, height, channel), dtype=np.float32)
    # y_batches = np.zeros((total_num_samples, width, height, target))
    # print(x_batches.shape, y_batches.shape)

    # TODO use pathlib
    ground_truth_set = glob.glob(os.path.join(y_fns, "*"))

        # Output
    output_data_dir = os.path.join(output_root, region)
    # if not os.path.exists(output_data_dir):
    # os.makedirs(output_data_dir, exist_ok=True)
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
    # partition_name = np.random.choice(['train', 'validation', 'test'], weights=[pct_train, pct_validation, 1-pct_train - pct_val])
        
    ground_truth_set = glob.glob(y_fns + "*")
    ground_truth_set = [x for x in ground_truth_set if region in x]

    count = 0
    non_zero_count = 0

    # total_num_samples is # training instances
    while count < total_num_samples:
        # Randomly choose a file from input list
        y_fn = np.random.choice(ground_truth_set)
        folder_name = y_fn.split('/')[2][2:7]
        filename = y_fn.split('/')[2][:26]
        x_fn = x_fns + filename + ".tif"
        # print(x_fn)

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
            # import IPython; import sys; IPython.embed(); sys.exit(1)
            # mask = target[y-(height//2):y+(height//2), x-(width//2):x+(width//2)]
            # mask = mask.reshape(width, height, 1)

            
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

    with open("./random_ratio_single.txt", "w+") as f:
        f.write("Ratio of chicken house to non-chicken: {}".format(non_zero_count / count))

    print("# of chicken house patches: {}".format(non_zero_count))
    # return x_batches, y_batches


# TODO, dump to data directory, not this directory
def gen_training_patches_balanced(x_fns, y_fns, width, height, channel, target, total_num_samples, region, output_root="./data/balanced", pct_train=0.80, pct_validation=0.15, seed=42):

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
    # if not os.path.exists(output_data_dir):
    # os.makedirs(output_data_dir, exist_ok=True)
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
    # partition_name = np.random.choice(['train', 'validation', 'test'], weights=[pct_train, pct_validation, 1-pct_train - pct_val])
        
    ground_truth_set = glob.glob(y_fns + "*")
    ground_truth_set = [x for x in ground_truth_set if region in x]

    count = 0
    non_zero_count = 0
    files_wo_chicken_houses = 0

    while count < total_num_samples:
        # Randomly choose a file from input list
        y_fn = np.random.choice(ground_truth_set)
        folder_name = y_fn.split('/')[2][2:7]
        filename = y_fn.split('/')[2][:26]
        x_fn = x_fns + filename + ".tif"
        # print(x_fn)

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
        # # if this is false, problem
        # if (len(x_ind) != 0):
        #     x_out_of_bound = (x_ind - width >= 0) | (x_ind + width < data.shape[1])
        #     y_out_of_bound = (y_ind - height >= 0) | (y_ind + height < data.shape[0])
        #     out_of_bounds = x_out_of_bound | y_out_of_bound
        #     y_ind = y_ind[~out_of_bounds]
        #     x_ind = x_ind[~out_of_bounds]
        #     files_wo_chicken_houses = 0
        # else:
        #     if files_wo_chicken_houses > 5:
        #         continue
        #     files_wo_chicken_houses += 1

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

            # remove this
            # x_batches[count] = img

            # 256 x 256 x (4 + 2)
            # data = np.random.randint(0, 255, (10,10)).astype(np.uint8)
            # im = Image.fromarray(data)
            # im.save('test.tif')

            # TODO store img in file for count
            
            # FOR DENSE
            mask = target_one_hot[y-(height//2):y+(height//2), x-(width//2):x+(width//2)]
            # mask = target[y-(height//2):y+(height//2), x-(width//2):x+(width//2)]
            # mask = mask.reshape(width, height, 1)

            # y_batches[count] = mask
            # {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}

            partition_name = all_partition_names[count]
            img_file = os.path.join(output_data_dir, partition_name, 'img', f"img_{count}")
            mask_file = os.path.join(output_data_dir, partition_name, 'mask', f"mask_{count}")
            np.save(img_file, img)
            np.save(mask_file, mask)

            # TODO store y_batches in mask file for count

            if (len(np.unique(mask, axis=0, return_counts=True)[1]) >= 2):
                # print(len(x_ind))
                # print(np.unique(y_batches[count], axis=0, return_counts=True)[0])
                non_zero_count += 1
            
            count += 1

            if count % 10000 == 0:
                print("Iteration: {}".format(count))

    # x_batches /= 255.0
    with open("./balanced_ratio_single.txt", "w+") as f:
        f.write("Ratio of chicken house to non-chicken: {}".format(non_zero_count / count))
    # print("Ratio of chicken house to non-chicken: {}".format(non_zero_count / count))
    print("# of chicken house patches: {}".format(non_zero_count))


    # return x_batches, y_batches


def main():
    # Sample 50k patches of 240x240 images
    # os.environ["DATA_DIR"]
    # in .bashrc: export DATA_DIR="asdf"

    data_root = "../../../data/jason/gen_data/balanced"
    random_data_root = "../../../data/jason/gen_datas/random"
    
    region = "m_38075"
    gen_training_patches("../../../data/jason/datasets/md_100cm_2017/38075/",
     "./binary_raster_md_tif/", 256, 256, 4, 2, 640, region="m_3807537_nw", output_root=data_root, pct_train=0.80, pct_validation=0.15, seed=42)


    gen_training_patches_balanced("../../../data/jason/datasets/md_100cm_2017/38075/",
     "./binary_raster_md_tif/", 256, 256, 4, 2, 640, region="m_3807537_nw", output_root=data_root, pct_train=0.80, pct_validation=0.15, seed=42)

 
    # we create two instances with the same arguments

    # train_generator = ChickenDataGenerator(
    #     dataset_dir=os.path.join(data_root, region, "train"),
    #     batch_size=2,
    #     steps_per_epoch=2,
        # )   

    # import IPython; import sys; IPython.embed(); sys.exit(1)
    pass

if __name__ == "__main__":
    main()
