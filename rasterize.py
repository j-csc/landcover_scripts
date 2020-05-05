import rasterio
import subprocess
f = rasterio.open("./all_test_inference_md/m_3807537_nw_18_1_20170611_inference.tif","r")
left, bottom, right, top = f.bounds
crs = f.crs.to_string()
height, width = f.height, f.width
f.close()
command = [
    "gdal_rasterize",
    "-ot", "Byte",
    "-burn", "1.0",
    "-of", "GTiff",
    "-te", str(left), str(bottom), str(right), str(top),
    "-ts", str(width), str(height),
    "-co", "COMPRESS=LZW",
    "-co", "BIGTIFF=YES",
    "./post_processed_md_inference/m_3807537_nw_18_1_20170611_inference_processed.geojson",
    "test_r.tif"
]
subprocess.call(command)