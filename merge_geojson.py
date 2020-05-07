import glob
import os, sys, argparse
import geopandas as gpd
import pandas as pd

def merge(input_dir, output_fn):
    list_of_files = glob.glob(input_dir + "*")
    list_of_gdfs = [gpd.read_file(fn) for fn in list_of_files]
    print(len(list_of_gdfs))
    final_gdf = pd.concat(list_of_gdfs)
    final_gdf.reset_index(inplace = True, drop = True) 
    final_gdf.to_file(output_fn, driver='GeoJSON')

def main():
    parser = argparse.ArgumentParser(description="Merge Geojson")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--input_dir", action="store", dest="input_dir", type=str, help="Path/paths to directory containing Geojson files", required=True)
    parser.add_argument("--output_fn", action="store", dest="output_fn", type=str, help="Output path for combined geojson", required=True)
    
    args = parser.parse_args(sys.argv[1:])

    merge(args.input_dir, args.output_fn)


if __name__ == "__main__":
    main()