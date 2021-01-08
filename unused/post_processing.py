import math
import numpy as np
import matplotlib.pyplot as plt
import fiona
import shapely
import shapely.geometry
import cv2
import rasterio
import rasterio.features
import geopandas as gpd
from scipy.signal import convolve2d, correlate2d, fftconvolve
from skimage.transform import rotate
from skimage.draw import circle
from skimage.io import imsave
import custom_loss
import argparse
import sys, os
import torch
import torch.nn

def post_processing(input_fn, area_thresh, aspect_ratio_thresh):
    data = None
    with rasterio.open(input_fn) as f:
        data = f.read().squeeze()
        data_transform = f.transform
        data_crs = f.crs
        
    mask = data == 2
    polygons = list(
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(rasterio.features.shapes(data, mask=mask, transform=data_transform))
    )

    print(len(polygons))
    
    # Filter by shape
    shape_area_threshold = area_thresh

    polygons_filtered_by_size = []
    for polygon in polygons:
        shape = shapely.geometry.shape(polygon["geometry"])

        if shape.area > shape_area_threshold:
            polygons_filtered_by_size.append(polygon)
    
    print(len(polygons_filtered_by_size))
            
    # Filter by aspect ratio
    aspect_ratio_threshold = aspect_ratio_thresh

    polygons_filtered_by_aspect_ratio = []
    poly_list = []
    for polygon in polygons_filtered_by_size:
        shape = shapely.geometry.shape(polygon["geometry"])

        side_lengths = custom_loss.get_side_lengths(shape.minimum_rotated_rectangle)    
        short_length = min(side_lengths)
        long_length = max(side_lengths)

        aspect_ratio = long_length / short_length
        if aspect_ratio > aspect_ratio_threshold:
            polygons_filtered_by_aspect_ratio.append(polygon)
            poly_list.append(shape)

    print(len(polygons_filtered_by_aspect_ratio))
            
    polys_gdf = gpd.GeoDataFrame(geometry=poly_list)
    list_of_polygons = []

    for i in range(len(polys_gdf)):
        plt.close('all')
        fig, ax = plt.subplots()
        ax.axis('off')
        polys_gdf.loc[[i],'geometry'].plot(ax = ax)
        fig.canvas.draw()
        arr = np.array(fig.canvas.renderer.buffer_rgba())[:,:,0]
        arr[arr == 255] = 0
        arr[arr == 31] = 1
        list_of_polygons.append(arr)
    
    return list_of_polygons, polygons_filtered_by_aspect_ratio, data_crs

def run_corr(pos_kernels, lop, polygons_filtered_by_aspect_ratio_and_size):
    filter_corrs_idx = []
    count = 0
    lop_final = []
    print(len(lop))
    for poly in lop:
        corrs_mean = []
        print(count)
        for i in range(len(pos_kernels)):
            corrs_mean.append(np.mean(fftconvolve(poly, pos_kernels[i])))
        # Valid ones
        print(np.max(corrs_mean))
        if np.max(corrs_mean) > 30:
            filter_corrs_idx.append(count)
        count += 1
    print(len(filter_corrs_idx))
    for i in range(len(lop)):
        if (i in filter_corrs_idx):
            lop_final.append(polygons_filtered_by_aspect_ratio_and_size[i])
    print(len(lop_final))
    return lop_final

def main():
    parser = argparse.ArgumentParser(description="Post processing script")
    # parser.add_argument("--shp", action="store", dest="shp_file", type=str, help="Input shapefile path (i.e. ../data/input.shp)", required=True)
    parser.add_argument("--input_fn", action="store", dest="input_fn", type=str, help="Input inference tif file (i.e. ../data/input.tif)", required=True)
    parser.add_argument("--output_fn", action="store", dest="output_fn", type=str, help="Output post-processed inference tif file (i.e. ../data/output.tif)", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", required=True)

    args = parser.parse_args(sys.argv[1:])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

    pos_kernels, neg_kernels, side_lengths, areas = custom_loss.get_pos_neg_kernel_and_more("../notebooks/Delmarva_PL_House_Final/Delmarva_PL_House_Final.shp")
    lop,polygons_filtered_by_aspect_ratio_and_size,crs = post_processing(args.input_fn, 525, 4)
    
    # Set up lop for torch
    torch_lop = []
    for p in lop:
        data_torch = torch.from_numpy(p[np.newaxis, np.newaxis, :, :].astype(np.float16)).to("cuda")
        data_torch.require_grad = False
        torch_lop.append(data_torch)

    pos_kernels = np.array(pos_kernels)
    pos_kernels = pos_kernels[:,np.newaxis,:,:]
    pos_kernels = torch.from_numpy(pos_kernels.astype(np.float16)).to("cuda")
    pos_kernels.require_grad = False

    neg_kernels = np.array(neg_kernels)
    neg_kernels = neg_kernels[:,np.newaxis,:,:]
    neg_kernels = torch.from_numpy(neg_kernels.astype(np.float16)).to("cuda")
    neg_kernels.require_grad = False

    filter_corrs_idx = []
    count = 0
    lop_final = []

    # Next step would be to filter by max activation, but rn this will suffice i hope

    for i in range(len(torch_lop)):
        outputs_pos = torch.nn.functional.conv2d(torch_lop[i], pos_kernels, padding=25)

        with torch.no_grad():
            outputs_pos = outputs_pos.cpu().numpy().astype(np.float32)
        outputs_pos = np.rollaxis(outputs_pos.squeeze(), 0, 3)
        torch.cuda.empty_cache()

        outputs_neg = torch.nn.functional.conv2d(1-data_torch, neg_kernels, padding=25)

        with torch.no_grad():
            outputs_neg = outputs_neg.cpu().numpy().astype(np.float32)
        outputs_neg = np.rollaxis(outputs_neg.squeeze(), 0, 3)
        torch.cuda.empty_cache()
        # print(outputs_pos.mean() - outputs_neg.mean())

        if (outputs_pos.mean()-outputs_neg.mean()) > 180:
            filter_corrs_idx.append(i)
        
    for i in range(len(lop)):
        if (i in filter_corrs_idx):
            lop_final.append(polygons_filtered_by_aspect_ratio_and_size[i])

    with fiona.open(
        args.output_fn, 'w',
        driver="GeoJSON",
        crs= crs,
        schema={'properties': [('raster_val', 'int')], 'geometry': 'Polygon'}
    ) as dst:
        dst.writerecords(lop_final)


if __name__ == "__main__":
    main()