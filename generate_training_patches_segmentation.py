import glob
import rasterio
import fiona
import numpy as np
import keras


# def gen_training_patches_single_center_and_dense(x_fns, y_fns, width, height, channel, target, batch_size, test=True):
#     # Output
#     x_batches = np.zeros((batch_size, width, height, channel), dtype=np.float32)
#     y_batches = np.zeros((batch_size, width, height, target))
#     y_batches_single = np.zeros((batch_size, width, height, target))
#     print(x_batches.shape, y_batches.shape)

#     ground_truth_set = glob.glob(y_fns + "*")

#     if (test == True):
#         ground_truth_set = [x for x in ground_truth_set if "m_3807537_ne" in x]

#     count = 0
#     non_zero_count = 0

#     # Randomly choose a file from input list
#     y_fn = np.random.choice(ground_truth_set)
#     folder_name = y_fn.split('/')[2][2:7]
#     filename = y_fn.split('/')[2][:26]
#     x_fn = x_fns + filename + ".mrf"
#     print(x_fn)

#     # Load input file
#     f = rasterio.open(x_fn, "r")
#     data = f.read().squeeze()
#     data = np.rollaxis(data, 0, 3)
#     f.close()
#     print(data.shape)

#     # Load ground truth file
#     f = rasterio.open(y_fn, "r")
#     target = f.read().squeeze()
#     f.close()
#     target_one_hot = keras.utils.to_categorical(target, num_classes=2)

#     # Poultry pixels
#     y_ind,x_ind = np.where(target==1)

#     # Used pixels
#     used_x = set()

#     # Randomly sample batch_size # of 240x240 patch per file
#     while count != batch_size:
#         x = np.random.randint(0, data.shape[1]-width)
#         y = np.random.randint(0, data.shape[0]-height)

#         # Force to choose chicken house pixel for ~50% of data
#         if len(x_ind) != 0 and count % 2 == 0:
#             rand_index = np.random.randint(0,len(x_ind))
#             x = x_ind[rand_index]
#             y = y_ind[rand_index]
#             temp_count = 0
#             while not (x-width >= 0 and x+width < data.shape[1] and y-height >= 0 and y+height < data.shape[0]):
#                 if (temp_count > 2):
#                     x = np.random.randint(0, data.shape[1]-width)
#                     y = np.random.randint(0, data.shape[0]-height)
#                 rand_index = np.random.randint(0,len(x_ind))
#                 x = x_ind[rand_index]
#                 y = y_ind[rand_index]
#                 temp_count += 1

#         while (not (x-width >= 0 and x+width < data.shape[1] and y-height >= 0 and y+height < data.shape[0])) or x in used_x:
#             x = np.random.randint(0, data.shape[1]-width)
#             y = np.random.randint(0, data.shape[0]-height)

#         used_x.add(x)

#         # Set up x_batch with img data at y,x coords
#         img = data[y-(height//2):y+(height//2), x-(width//2):x+(width//2), :].astype(np.float32)
#         x_batches[count] = img
        
#         # FOR DENSE
#         y_batches[count] = target_one_hot[y-(height//2):y+(height//2), x-(width//2):x+(width//2)]

#         # FOR SINGLE
#         # Center predict given context
#         label = target[y,x]

#         y_batches_single[count,height//2,width//2,label] = 1

#         if (label != 0):
#             non_zero_count += 1
        
#         count += 1

#         if count % 100 == 0:
#             print(f"Iteration: {count}")

#     x_batches = x_batches/255.0
#     y_batches = keras.utils.to_categorical(y_batches, num_classes=2)

#     print(y_batches.shape)

#     print("Ratio of chicken house to non-chicken: {}".format(non_zero_count / count))
#     print("# of chicken house patches: {}".format(non_zero_count))
#     np.save('./x_dense.npy',x_batches)
#     np.save('./y_dense.npy',y_batches)
#     np.save('./y_single.npy', y_batches_single)

#     return x_batches, y_batches

def gen_training_patches_center_and_dense_single(x_fns, y_fns, width, height, channel, target, batch_size, test=True):
    # Output
    x_batches = np.zeros((batch_size, width, height, channel), dtype=np.float32)
    y_batches = np.zeros((batch_size, width, height, target))
    y_batches_single = np.zeros((batch_size, width, height, target))
    print(x_batches.shape, y_batches.shape)

    ground_truth_set = glob.glob(y_fns + "*")

    if (test == True):
        ground_truth_set = [x for x in ground_truth_set if "m_3807537_ne" in x]

    count = 0
    non_zero_count = 0

    # Used pixels
    used_x = set()

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

    # Get one-hot-encoding for ground truth
    target_one_hot = keras.utils.to_categorical(target, num_classes=2)

    # Poultry pixels
    y_ind,x_ind = np.where(target==1)

    # Randomly sample patch_size # of 240x240 patch per file
    while count < batch_size:
        x = np.random.randint(width, data.shape[1]-width)
        y = np.random.randint(height, data.shape[0]-height)

        # Force to choose chicken house pixel for ~50% of data
        if len(x_ind) != 0 and count % 2 == 0:
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
                if (temp_count > 2):
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

        # FOR SINGLE
        # Center predict given context
        label = target[y,x]

        y_batches_single[count,height//2,width//2,label] = 1

        if (label != 0):
            non_zero_count += 1
        
        count += 1

        if count % 1000 == 0:
            print(f"Iteration: {count}")

    x_batches = x_batches/255.0
    # # y_batches = keras.utils.to_categorical(y_batches, num_classes=2)

    print(y_batches.shape)

    print("Ratio of chicken house to non-chicken: {}".format(non_zero_count / count))
    print("# of chicken house patches: {}".format(non_zero_count))
    # np.save('./x_dense.npy',x_batches)
    # np.save('./y_dense.npy',y_batches)
    # np.save('./y_single.npy', y_batches_single)

    return x_batches, y_batches

def gen_training_patches_center_and_dense(x_fns, y_fns, width, height, channel, target, batch_size, test=False):
    # Output
    x_batches = np.zeros((batch_size, width, height, channel), dtype=np.float32)
    y_batches = np.zeros((batch_size, width, height, target))
    y_batches_single = np.zeros((batch_size, width, height, target))
    print(x_batches.shape, y_batches.shape)

    ground_truth_set = glob.glob(y_fns + "*")

    if (test == True):
        ground_truth_set = [x for x in ground_truth_set if "m_39075" in x]

    count = 0
    non_zero_count = 0

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
        print(data.shape)

        # Load ground truth file
        f = rasterio.open(y_fn, "r")
        target = f.read().squeeze()
        f.close()

        # Get one-hot-encoding for ground truth
        target_one_hot = keras.utils.to_categorical(target, num_classes=2)

        # Poultry pixels
        y_ind,x_ind = np.where(target==1)

        # Used pixels
        used_x = set()

        # Randomly sample patch_size # of 240x240 patch per file
        for i in range(100):
            x = np.random.randint(width, data.shape[1]-width)
            y = np.random.randint(height, data.shape[0]-height)

            # Force to choose chicken house pixel for ~50% of data
            if len(x_ind) != 0 and count % 2 == 0:
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
                    if (temp_count > 2):
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

            # FOR SINGLE
            # Center predict given context
            label = target[y,x]

            y_batches_single[count,height//2,width//2,label] = 1

            if (label != 0):
                non_zero_count += 1
            
            count += 1

            if count % 1000 == 0:
                print(f"Iteration: {count}")

    x_batches = x_batches/255.0
    # # y_batches = keras.utils.to_categorical(y_batches, num_classes=2)

    print(y_batches.shape)

    print("Ratio of chicken house to non-chicken: {}".format(non_zero_count / count))
    print("# of chicken house patches: {}".format(non_zero_count))
    # np.save('./x_dense_val.npy',x_batches)
    # np.save('./y_dense_val.npy',y_batches)
    # np.save('./y_single.npy', y_batches_single)

    return x_batches, y_batches

def main():
    _,_ = gen_training_patches_center_and_dense("../../../media/disk2/datasets/all_maryalnd_naip/",
     "./binary_raster_md_tif/", 150, 150, 4, 2, 10000, test=True)

    # print(y)
    
    pass

if __name__ == "__main__":
    main()