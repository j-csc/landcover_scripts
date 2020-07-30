import glob
import rasterio
import fiona
import numpy as np

def gen_training_patches_single_center_pixel(x_fns, y_fns, width, height, channel, target, batch_size, test=True):
    # Output
    x_batches = np.zeros((batch_size, width, height, channel), dtype=np.float32)
    y_batches = np.zeros((batch_size, width, height, target))
    print(x_batches.shape, y_batches.shape)

    ground_truth_set = glob.glob(y_fns + "*")

    if (test == True):
        ground_truth_set = [x for x in ground_truth_set if "m_3807537_ne" in x]

    count = 0
    non_zero_count = 0

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
    print(data.shape)

    # Load ground truth file
    f = rasterio.open(y_fn, "r")
    target = f.read().squeeze()
    f.close()

    # Poultry pixels
    y_ind,x_ind = np.where(target==1)

    # Randomly sample batch_size # of 240x240 patch per file
    while count != batch_size:
        x = np.random.randint(0, data.shape[1]-width)
        y = np.random.randint(0, data.shape[0]-height)

        # # Force to choose chicken house pixel
        # if (len(np.unique(target)) == 2) and (len(np.unique((target[y:y+height, x:x+width]))) != 2):
            
        #     rand_index = np.random.randint(0,len(x_ind))
        #     x = x_ind[rand_index]
        #     y = y_ind[rand_index]
        #     temp_count = 0
        #     while not (x-width >= 0 and x+width < data.shape[1] and y-height >= 0 and y+height < data.shape[0]):
        #         if (temp_count > 2):
        #             x = np.random.randint(0, data.shape[1]-width)
        #             y = np.random.randint(0, data.shape[0]-height)
        #         rand_index = np.random.randint(0,len(x_ind))
        #         x = x_ind[rand_index]
        #         y = y_ind[rand_index]
        #         temp_count += 1

        while not (x-width >= 0 and x+width < data.shape[1] and y-height >= 0 and y+height < data.shape[0]):
            x = np.random.randint(0, data.shape[1]-width)
            y = np.random.randint(0, data.shape[0]-height)

        # MAKE x,y THE CENTER

        # Set up x_batch with img data at y,x coords
        img = data[y-(height/2):y+(height/2), x-(width/2):x+(width/2), :].astype(np.float32)
        x_batches[count] = img
        
        # Center predict given context
        label = target[y,x]

        y_batches[count,75,75,label] = 1

        if (label != 0):
            non_zero_count += 1
        
        count += 1

        if count % 1000 == 0:
            print(f"Iteration: {count}")

    x_batches = x_batches/255.0
    # y_batches = keras.utils.to_categorical(y_batches, num_classes=2)

    print(y_batches.shape)

    print("Ratio of chicken house to non-chicken: {}".format(non_zero_count / count))

    np.save('./xtrain_single.npy',x_batches)
    np.save('./ytrain_single.npy',y_batches)

    return x_batches, y_batches

def gen_training_patches_single_dense(x_fns, y_fns, width, height, channel, target, batch_size, test=True):
    # Output
    x_batches = np.zeros((batch_size, width, height, channel), dtype=np.float32)
    y_batches = np.zeros((batch_size, width, height, target))
    print(x_batches.shape, y_batches.shape)

    ground_truth_set = glob.glob(y_fns + "*")

    if (test == True):
        ground_truth_set = [x for x in ground_truth_set if "m_3807537_ne" in x]

    count = 0
    non_zero_count = 0

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
    print(data.shape)

    # Load ground truth file
    f = rasterio.open(y_fn, "r")
    target = f.read().squeeze()
    f.close()

    # Poultry pixels
    y_ind,x_ind = np.where(target==1)

    # Randomly sample batch_size # of 240x240 patch per file
    while count != batch_size:
        x = np.random.randint(0, data.shape[1]-width)
        y = np.random.randint(0, data.shape[0]-height)

        # # Force to choose chicken house pixel
        # if (len(np.unique(target)) == 2) and (len(np.unique((target[y:y+height, x:x+width]))) != 2):
            
        #     rand_index = np.random.randint(0,len(x_ind))
        #     x = x_ind[rand_index]
        #     y = y_ind[rand_index]
        #     temp_count = 0
        #     while not (x-width >= 0 and x+width < data.shape[1] and y-height >= 0 and y+height < data.shape[0]):
        #         if (temp_count > 2):
        #             x = np.random.randint(0, data.shape[1]-width)
        #             y = np.random.randint(0, data.shape[0]-height)
        #         rand_index = np.random.randint(0,len(x_ind))
        #         x = x_ind[rand_index]
        #         y = y_ind[rand_index]
        #         temp_count += 1

        while not (x-width >= 0 and x+width < data.shape[1] and y-height >= 0 and y+height < data.shape[0]):
            x = np.random.randint(0, data.shape[1]-width)
            y = np.random.randint(0, data.shape[0]-height)

        # MAKE x,y THE CENTER

        # Set up x_batch with img data at y,x coords
        img = data[y-75:y+74+1, x-75:x+74+1, :].astype(np.float32)
        x_batches[count] = img
        
        # Center predict given context
        for i in range
        label = target[y,x]

        y_batches[count,75,75,label] = 1

        if (label != 0):
            non_zero_count += 1
        
        count += 1

        if count % 1000 == 0:
            print(f"Iteration: {count}")

    x_batches = x_batches/255.0
    # y_batches = keras.utils.to_categorical(y_batches, num_classes=2)

    print(y_batches.shape)

    print("Ratio of chicken house to non-chicken: {}".format(non_zero_count / count))

    # np.save('./xtrain.npy',x_batches)
    # np.save('./ytrain.npy',y_batches)

    return x_batches, y_batches

def main():
    # Sample 50k patches of 240x240 images
    
    x,y = gen_training_patches_single_dense("../../../media/disk2/datasets/all_maryalnd_naip/",
     "./binary_raster_md_tif/", 150, 150, 4, 2, 1000)

    # print(y)
    
    pass

if __name__ == "__main__":
    main()