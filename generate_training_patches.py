import glob
import rasterio
import fiona
import numpy as np
import keras.utils

def gen_training_patches(x_fns, y_fns, width, height, channel, target, batch_size, loc='south'):
    # Output
    x_batches = np.zeros((batch_size, width, height, channel), dtype=np.float32)
    y_batches = np.zeros((batch_size, width, height, target+1), dtype=np.float32)

    y_batches[:,:,:] = [1] + [0] * (y_batches.shape[-1]-1)

    print(x_batches.shape, y_batches.shape)

    ground_truth_set = glob.glob(y_fns + "*")

    count = 0
    non_zero_count = 0

    # Optional, for south, north maryland check
    if loc == 'south':
        ground_truth_set = [i for i in ground_truth_set if '38075' in i or '38076' in i or '38077' in i]
    else:
        ground_truth_set = [i for i in ground_truth_set if '39075' in i or '39076' in i or '39077' in i]

    while count < batch_size:
        # Randomly choose a file from input list
        y_fn = np.random.choice(ground_truth_set)
        folder_name = y_fn.split('/')[2][2:7]
        filename = y_fn.split('/')[2][:26]
        x_fn = x_fns + filename + ".mrf"
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

        # Poultry pixels
        y_ind,x_ind = np.where(target==1)

        # Randomly sample 100 240x240 patch per file
        for i in range(100):
            if count != batch_size:
                x = np.random.randint(0, data.shape[1]-width)
                y = np.random.randint(0, data.shape[0]-height)

                # Force to choose chicken house pixel
                if (len(np.unique(target)) == 2) and (len(np.unique((target[y:y+height, x:x+width]))) != 2):
                    rand_index = np.random.randint(0,len(x_ind))
                    x = x_ind[rand_index]
                    y = y_ind[rand_index]
                    temp_count = 0
                    while not (x >= 0 and x < data.shape[1]-width and y >= 0 and y < data.shape[0]-height):
                        if (temp_count > 3):
                            x = np.random.randint(0, data.shape[1]-width)
                            y = np.random.randint(0, data.shape[0]-height)
                        rand_index = np.random.randint(0,len(x_ind))
                        x = x_ind[rand_index]
                        y = y_ind[rand_index]
                        temp_count += 1
                
                # Set up x_batch with img data at y,x coords
                img = data[y:y+height, x:x+width, :].astype(np.float32)
                x_batches[count] = img
                
                # Center predict given context
                label = target[y,x]
                
                y_batches[count,75,75,0] = 0
                y_batches[count,75,75,label+1] = 1

                if (label != 0):
                    non_zero_count += 1
                
                count += 1

                if count % 1000 == 0:
                    print(f"Iteration: {count}")

    x_batches = x_batches/255.0
    # y_batches = keras.utils.to_categorical(y_batches, num_classes=2)

    print(y_batches.shape)

    print(non_zero_count / count)

    np.save(x_batches, "x_train.npy")
    np.save(y_batches, "y_train.npy")

    return x_batches, y_batches

def main():
    # Sample 50k patches of 240x240 images
    
    x,y = gen_training_patches("../../../media/disk2/datasets/all_maryalnd_naip/",
     "./binary_raster_md_tif/", 150, 150, 4, 2, 10000)

    # print(y)
    
    pass

if __name__ == "__main__":
    main()