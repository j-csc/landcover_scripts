import glob
import rasterio
import fiona
import numpy as np
import keras.utils

def gen_training_patches(x_fns, y_fns, width, height, channel, target, batch_size):
    # Output
    x_batches = np.zeros((batch_size, width, height, channel), dtype=np.float32)
    y_batches = np.zeros((batch_size, width, height, target), dtype=np.float32)

    y_batches[:,:,:] = [1] + [0] * (y_batches.shape[-1]-1)

    print(x_batches.shape, y_batches.shape)

    ground_truth_set = glob.glob(y_fns + "*")

    count = 0
    non_zero_count = 0

    while count < batch_size:
        # Randomly choose a file from input list
        y_fn = np.random.choice(ground_truth_set)
        folder_name = y_fn.split('/')[2][2:7]
        filename = y_fn.split('/')[2][:26]
        x_fn = x_fns + folder_name + "/" + filename + ".mrf"

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
        # print(target.shape)

        # Randomly sample 100 240x240 patch per file
        for i in range(1):
            if count != batch_size:
                x = np.random.randint(0, data.shape[1]-width)
                y = np.random.randint(0, data.shape[0]-height)
                while np.any((data[y:y+height, x:x+width, :] == 0).sum(axis=2) == data.shape[2]):
                    x = np.random.randint(0, data.shape[1]-width)
                    y = np.random.randint(0, data.shape[0]-height)
                
                # Set up x_batch with img data at y,x coords
                img = data[y:y+height, x:x+width, :].astype(np.float32)
                x_batches[count] = img
                
                # Set up target labels with 4 dim
                temp_target = target[y:y+height, x:x+width]
                
                for j in range(0, height):
                    for k in range(0, width):
                        label = target[j,k]
                        y_batches[count,j,k,0] = 0
                        y_batches[count,j,k,label] = 1

                # print(np.unique(temp_target, return_counts=True))

                if (np.count_nonzero(temp_target) != 0):
                    non_zero_count += 1
                
                count += 1

                if count % 1000 == 0:
                    print(f"Iteration: {count}")

    x_batches = x_batches/255.0
    # y_batches = keras.utils.to_categorical(y_batches, num_classes=2)

    print(y_batches.shape)

    print(non_zero_count / count)

    return x_batches, y_batches

def main():
    # Sample 50k patches of 240x240 images
    gen_training_patches("../../../media/disk2/datasets/maaryland_naip_2017/",
     "./binary_raster_md_tif/", 240, 240, 4, 2, 2)
    
    pass

if __name__ == "__main__":
    main()