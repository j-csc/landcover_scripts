import glob
import rasterio
import fiona
import numpy as np
import keras.utils

def gen_training_patches(x_fns, y_fns, width, height, channel, target, batch_size, region=None):
    # Output
    x_batches = np.zeros((batch_size, width, height, channel), dtype=np.float32)
    y_batches = np.zeros((batch_size, width, height, target))
    print(x_batches.shape, y_batches.shape)

    ground_truth_set = glob.glob(y_fns + "*")

    if (region):
        ground_truth_set = [x for x in ground_truth_set if region in x]

    count = 0
    non_zero_count = 0

    while count < batch_size:
        # Randomly choose a file from input list
        y_fn = np.random.choice(ground_truth_set)
        folder_name = y_fn.split('/')[2][2:7]
        filename = y_fn.split('/')[2][:26]
        x_fn = x_fns + filename + ".tif"
        print(x_fn)

        # Load input file
        f = rasterio.open(x_fn, "r")
        data = f.read().squeeze()
        data = np.rollaxis(data, 0, 3)
        f.close()
        print(data.shape)

        # Load ground truth file
        f = rasterio.open(y_fn, "r")
        target = f.read().squeeze()
        f.close()

        # Get one-hot-encoding for ground truth
        target_one_hot = keras.utils.to_categorical(target, num_classes=2)
        # Used pixels
        used_x = set()

        # Randomly sample patch_size # of 240x240 patch per file
        for i in range(100):
            x = np.random.randint(width, data.shape[1]-width)
            y = np.random.randint(height, data.shape[0]-height)

            # Reselect until x is new pixel
            while x in used_x:
                x = np.random.randint(width, data.shape[1]-width)
                y = np.random.randint(height, data.shape[0]-height)

            used_x.add(x)

            # Set up x_batch with img data at y,x coords
            img = data[y-(height//2):y+(height//2), x-(width//2):x+(width//2), :].astype(np.float32)
            x_batches[count] = img
            
            # FOR DENSE
            y_batches[count] = target_one_hot[y-(height//2):y+(height//2), x-(width//2):x+(width//2)]

            if (len(np.unique(y_batches[count], return_counts=True)[0]) == 2):
                non_zero_count += 1
            
            count += 1

            if count % 1000 == 0:
                print("Iteration: {}".format(count))

    x_batches = x_batches/255.0

    print(y_batches.shape)

    print("Ratio of chicken house to non-chicken: {}".format(non_zero_count / count))
    print("# of chicken house patches: {}".format(non_zero_count))
    np.save('./x_dense_new.npy',x_batches)
    np.save('./y_dense_new.npy',y_batches)
    # np.save('./y_single.npy', y_batches_single)

    return x_batches, y_batches

def main():
    # Sample 50k patches of 240x240 images
    
    x,y = gen_training_patches("../../../media/disk2/datasets/all_maryalnd_naip/",
     "./binary_raster_md_tif/", 150, 150, 4, 2, 1000)

    # print(y)
    
    pass

if __name__ == "__main__":
    main()