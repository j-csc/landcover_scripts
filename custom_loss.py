import math
import numpy as np
import fiona
import shapely
import shapely.geometry
import cv2
import time
import argparse
import sys, os
from skimage.transform import rotate
from skimage.draw import circle
from skimage.io import imsave

def distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def get_side_lengths(shape):
    xs, ys = shape.boundary.xy
    pts = list(zip(xs, ys))
    lengths = []
    for i in range(len(pts)-1):
        lengths.append(distance(pts[i], pts[i+1]))
    assert len(lengths) == 4
    return sorted(lengths)

def generate_rectangle(aspect_ratio, height, width, rotation, min_pixel=100):
    redo = True
    img = np.zeros((width, height), dtype=np.uint8)
    while redo == True:
        start_x = np.random.randint(width//2)
        start_y = np.random.randint(height//2)

        # Chooses from short side...
        short_side = np.random.randint(height//3)
        long_side  = short_side * aspect_ratio

        end_y = int(start_y + short_side)
        end_x = int(start_x + long_side)

        # Get center
        center_x = (max(start_x,end_x) - min(start_x, end_x)) / 2
        center_y = (max(start_y,end_y) - min(start_y, end_y)) / 2

        rec = cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (10,10,10), -1)
        img = rotate(image=img, angle=rotation)

        area = short_side * long_side
        percent_area = area / (height*width) 
        
        # Make sure there's no empty images (at least 100 pixels are in img)
        if np.count_nonzero(img) < (min_pixel) or percent_area < 0.1:
            # reset
            redo = True
            img = np.zeros((width, height), dtype=np.uint8)
        else:
            redo = False
                    
    return img

def generate_circle(x,y,r,height,width):
    img = np.zeros((width, height), dtype=np.uint8)
    circle = cv2.circle(img, (x,y), r, (10, 10, 10), -1) 
    return img

def get_pos_neg_kernel(shp_file):
    f = fiona.open(shp_file,"r")
    total_num = 0
    valid_num = 0

    side_lengths = []
    areas = []
    for row in f:
        if row["geometry"]["type"] == "Polygon":
            shape = shapely.geometry.shape(row["geometry"])
            num_points = len(row["geometry"]["coordinates"][0])
            if num_points == 5:
                side_lengths.append(get_side_lengths(shape))
                areas.append(shape.area)
                valid_num += 1
        total_num += 1
    f.close()

    side_lengths = np.array(side_lengths)
    short_sides = side_lengths[:,:2].mean(axis=1)
    long_sides = side_lengths[:,2:].mean(axis=1)

    # Obtain aspect ratio
    aspect_ratios = long_sides / short_sides
    print("Average aspect ratio: {}".format(aspect_ratios.mean()))

    # Obtain positive kernel library
    aspect_ratio_fn = lambda: np.random.normal(loc=9.07, scale=1.71)
    pos_kernel = []
    for i in range(36):
        rect = generate_rectangle(aspect_ratio_fn(), 640, 480, np.random.randint(360), 800)
        pos_kernel.append(rect)

    # Obtain negative kernel library
    neg_kernel = []
    for i in range(36):
        circle = generate_circle(np.random.randint(640), np.random.randint(480), np.random.randint(100,480), 640, 480)
        neg_kernel.append(circle)
        
    print(neg_kernel[1].shape, pos_kernel[1].shape)

    return pos_kernel, neg_kernel

def get_pos_neg_kernel_and_more(shp_file):
    f = fiona.open(shp_file,"r")
    total_num = 0
    valid_num = 0

    side_lengths = []
    areas = []
    for row in f:
        if row["geometry"]["type"] == "Polygon":
            shape = shapely.geometry.shape(row["geometry"])
            num_points = len(row["geometry"]["coordinates"][0])
            if num_points == 5:
                side_lengths.append(get_side_lengths(shape))
                areas.append(shape.area)
                valid_num += 1
        total_num += 1
    f.close()

    side_lengths = np.array(side_lengths)
    short_sides = side_lengths[:,:2].mean(axis=1)
    long_sides = side_lengths[:,2:].mean(axis=1)

    # Obtain aspect ratio
    aspect_ratios = long_sides / short_sides
    print("Average aspect ratio: {}".format(aspect_ratios.mean()))

    # Obtain positive kernel library
    aspect_ratio_fn = lambda: np.random.normal(loc=9.07, scale=1.71)
    pos_kernels = []
    for i in range(36):
        rect = generate_rectangle(aspect_ratio_fn(), 640, 480, np.random.randint(360), 800)
        pos_kernels.append(rect)

    # Obtain negative kernel library
    neg_kernels = []
    for pos_kernel in pos_kernels:
        neg_kernel = 1 - pos_kernel.copy()
        neg_kernel = neg_kernel / (neg_kernel.sum() + 0.00001)
        neg_kernels.append(neg_kernel)
        
    return pos_kernels, neg_kernels, side_lengths,areas 

def main():
    parser = argparse.ArgumentParser(description="Generate custom shape loss")
    parser.add_argument("--shp", action="store", dest="shp_file", type=str, help="Input shapefile path (i.e. ../data/input.shp)", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", required=True)

    args = parser.parse_args(sys.argv[1:])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

    start_time = float(time.time())

    # Read in shapefile
    f = fiona.open(args.shp_file,"r")
    total_num = 0
    valid_num = 0

    side_lengths = []
    areas = []
    for row in f:
        if row["geometry"]["type"] == "Polygon":
            shape = shapely.geometry.shape(row["geometry"])
            num_points = len(row["geometry"]["coordinates"][0])
            if num_points == 5:
                side_lengths.append(get_side_lengths(shape))
                areas.append(shape.area)
                valid_num += 1
        total_num += 1
    f.close()

    side_lengths = np.array(side_lengths)
    short_sides = side_lengths[:,:2].mean(axis=1)
    long_sides = side_lengths[:,2:].mean(axis=1)

    # Obtain aspect ratio
    aspect_ratios = long_sides / short_sides
    print("Average aspect ratio: {}".format(aspect_ratios.mean()))

    # Obtain positive kernel library
    pos_kernel = []
    for i in range(10):
        rect = generate_rectangle(np.random.choice(aspect_ratios), 50, 50, np.random.randint(360))
        pos_kernel.append(rect)

    # Obtain negative kernel library
    neg_kernel = []
    for i in range(10):
        circle = generate_circle(np.random.randint(50), np.random.randint(50), np.random.randint(50*2), 50, 50)
        neg_kernel.append(circle)



    pass
