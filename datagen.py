import keras
import numpy as np
import os
import glob
import rasterio

class ChickenDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        dataset_dir, # e.g. os.path.join(data_root, region, "train")
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
            y_batch[i] = mask_data

        return x_batch, y_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.img_names))
        if self.shuffle:
            np.random.shuffle(self.indices)

# Datagen for experiments

# TODO, dump to data directory, not this directory

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
        folder_name = y_fn.split('/')[2][2:7]
        filename = y_fn.split('/')[2][:26]
        x_fn = x_fns + f"/{folder_name}/" + filename + ".tif"
        
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
            img_file = os.path.join(output_data_dir, partition_name, 'img', f"img_{count}")
            mask_file = os.path.join(output_data_dir, partition_name, 'mask', f"mask_{count}")
            np.save(img_file, img)
            np.save(mask_file, mask)
            # in __getitem__ index, read file from index and return result
            count+=1

def get_patches_balanced(ground_truth_set, x_fns, output_data_dir, width, height, total_num_samples, partition_name):
    count = 0
    # Get training set
    while count < total_num_samples:
        # Randomly choose a file from input list
        y_fn = np.random.choice(ground_truth_set)
        folder_name = y_fn.split('/')[2][2:7]
        filename = y_fn.split('/')[2][:26]
        x_fn = x_fns + f"/{folder_name}/" + filename + ".tif"
        
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

        # Poultry pixels
        y_ind,x_ind = np.where(target==1)
        
       
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

            # FOR DENSE
            mask = target_one_hot[y-(height//2):y+(height//2), x-(width//2):x+(width//2)]
            
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

# TODO, dump to data directory, not this directory
def gen_training_patches_balanced_sep(x_fns, y_fns, width, height, channel, target, total_num_samples, region, output_root, train_reg, test_reg, val_reg, pct_train=0.80, pct_validation=0.15, seed=42):

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
        
    ground_truth_set = glob.glob(y_fns + "*")

    train_set, test_set, val_set = get_gt_set(train_reg, test_reg, val_reg, ground_truth_set)
    print(num_train, num_test, num_validation) 

    get_patches_balanced(train_set, x_fns, output_data_dir, width, height, num_train, "train")
    get_patches_balanced(test_set, x_fns, output_data_dir, width, height, num_test, "test")
    get_patches_balanced(val_set, x_fns, output_data_dir, width, height, num_validation, "validation")

    pass



def main():
    # Sample 50k patches of 240x240 images
    # os.environ["DATA_DIR"]
    # in .bashrc: export DATA_DIR="asdf"

    data_root = "../../../data/jason/gen_data/balanced"
    random_data_root = "../../../data/jason/gen_data/random"
    region = "exp2"
    
    # region = "m_38075"
    region_dict = {
            "poultry_region_1": "m_38075", 
            "poultry_region_2": "m_38076",
            "poultry_region_3": "m_39075",
            "non_poultry_region_1": "m_39077",
            "non_poultry_region_2": "m_38077"
        }

    train_reg = [region_dict["poultry_region_1"]]
    test_reg = [region_dict["poultry_region_2"]]
    val_reg = [region_dict["poultry_region_3"]]

    gen_training_patches_sep("../../../data/jason/datasets/md_100cm_2017",
     "./binary_raster_md_tif/", 256, 256, 4, 2, 100000, region=region, output_root=random_data_root, train_reg=train_reg, test_reg=test_reg, val_reg=val_reg,  pct_train=0.80, pct_validation=0.15, seed=42)

    gen_training_patches_balanced_sep("../../../data/jason/datasets/md_100cm_2017",
        "./binary_raster_md_tif/", 256, 256, 4, 2, 100000, region=region, output_root=data_root, train_reg=train_reg, test_reg=test_reg, val_reg=val_reg,  pct_train=0.80, pct_validation=0.15, seed=42)

    # import IPython; import sys; IPython.embed(); sys.exit(1)
    pass

if __name__ == "__main__":
    main()
