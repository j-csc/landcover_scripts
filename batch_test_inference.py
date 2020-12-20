import subprocess
import glob
import sys, os, time
from multiprocessing import Process, Queue
import rasterio
 
def do_work(work, gpu_idx):
    while not work.empty():
        fn = work.get()
        filename = (fn.split("/")[-1].split(".")[0])
        print(fn)

        #################
        # Generate index
        #################

        # test_index_fn = "./test_index_de/" + filename + "_index.geojson"
        # subprocess.call(["gdaltindex", 
        #     "-t_srs", "epsg:32618",
        #     "-f", "GeoJSON",
        #     test_index_fn,
        #     fn])


        #################
        # Get raster md
        #################

        # shp_out_fn = "./binary_raster_de/" + filename + "_rasterized.geojson"
        # subprocess.call(["ogr2ogr", 
        #     "-clipsrc", test_index_fn,
        #     shp_out_fn,
        #     "../notebooks/Delmarva_PL_House_Final/Delmarva_PL_House_Final.shp"])


        #################
        # Test Inference
        #################

        out_fn = f"./test_results/exp1_random/{filename}_random.tif"
        subprocess.call(["python3","./test_inference.py",
            "--input_fns", fn,
            "--output_fns", out_fn,
            "--model", "./unet_model_random.h5",
            "--gpu", str(gpu_idx)])

        # subprocess.call(["python","./post_processing.py",
        #     "--input_fn", fn,
        #     "--output_fn", out_fn,
        #     "--gpu", str(gpu_idx)])

        #################
        # Get raster de
        #################

        # p_fn = f"./binary_raster_de/{filename}_rasterized.geojson"
        # out_fn = f"./binary_raster_de_tif/{filename}_rasterized.tif"
        # f = rasterio.open(fn,"r")
        # left, bottom, right, top = f.bounds
        # crs = f.crs.to_string()
        # height, width = f.height, f.width
        # f.close()
        # command = [
        #     "gdal_rasterize",
        #     "-ot", "Byte",
        #     "-burn", "1.0",
        #     "-of", "GTiff",
        #     "-te", str(left), str(bottom), str(right), str(top),
        #     "-ts", str(width), str(height),
        #     "-co", "COMPRESS=LZW",
        #     "-co", "BIGTIFF=YES",
        #     p_fn,
        #     out_fn
        # ]
        # subprocess.call(command)
        
    return True

def batch_run(ALL_FNS):
    work = Queue()
    GPUS = [0,1,2,3]
    for fn in ALL_FNS:
        work.put(fn)
    processes = []
    for gpu_idx in GPUS:
        p = Process(target=do_work, args=(work, gpu_idx))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

def main():
    # Batch run inference
    # fn_folders = glob.glob("../../../media/disk1/datasets/delaware_data/de_100cm_2017/*") # Delaware
    
    fn_folders = glob.glob("../../../media/data/datasets/md/md_100cm_2017/38075/*.tif") # MD
    
    all_fns = []
    for fn_folder in fn_folders:
        fns = (glob.glob(fn_folder + "/*.tif"))
        print(fns)
        for fn in fns:
            all_fns.append(fn)
    # print(fn_folders)


    batch_run(fn_folders)

    # Batch run post-processing
    # fn_folders = glob.glob("./binary_raster_md/*")
    # fn_folders = glob.glob("./all_test_inference_de/*")
    # batch_run(fn_folders)

    # Rasterize
    # gdaltindex -t_srs epsg:32618 -f GeoJSON md_test_index.geojson ./post_processed_de_tif/m_3807537_nw_18_1_20170720_inference_processed.tif
    # ogr2ogr -clipsrc md_test_index.geojson md_test_clipped.shp ../notebooks/Delmarva_PL_House_Final/Delmarva_PL_House_Final.shp

    # Small batch inference
    # exps = (glob.glob("./test_run/single_tile_4000s/*"))
    # count = 0
    # for e in exps:
    #     count += 1
    #     # print(e)
    #     output_name = (e.split("/")[-1])[:-8]
    #     print(output_name)
    #     subprocess.call(["python","./test_inference.py",
    #                 "--input_fns", "../../../media/disk2/datasets/all_maryalnd_naip/m_3807708_sw_18_1_20170716.mrf",
    #                 "--output_fns", f"./test_run/single_tile_4000s_inf_3/{output_name}_single_inference.tif",
    #                 "--model", e,
    #                 "--gpu", "2"])
    # subprocess.call(["python","./test_inference.py",
    #             "--input_fns", "../../../media/disk2/datasets/maaryland_naip_2017/38075/m_3807536_se_18_1_20170611.mrf",
    #             "--output_fns", "./m_3807536_se_18_1_20170611_sup_uneven_best.tif",
    #             "--model", "./new/experiment9/tmp_sup_uneven/sup_tuned_model_uneven_03_0.24.h5",
    #             "--gpu", "1"])
    # subprocess.call(["python","./test_inference.py",ss
    #             "--input_fns", "../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf",
    #             "--output_fns", "./m_3807537_ne_18_1_20170611_ae_even_best.tif",
    #             "--model", "./new/experiment10/tmp_ae_even/ae_tuned_model_even_08_0.17.h5",
    #             "--gpu", "1"])
    # subprocess.call(["python","./test_inference.py",
    #             "--input_fns", "../../../media/disk2/datasets/maaryland_naip_2017/38075/m_3807536_se_18_1_20170611.mrf",
    #             "--output_fns", "./m_3807536_se_18_1_20170611_ae_uneven_best.tif",
    #             "--model", "./new/experiment3/tmp_ae_uneven/ae_tuned_model_uneven_03_0.20.h5",
    #             "--gpu", "1"])

    # Generate model
    # for e in exps:
    #     print(e)
        # Balanced
        # subprocess.call(["python", "./generate_tuned_model_v2.py",
        # "--in_geo_path", "../notebooks/all_corrections_no_dups.geojson",
        # "--in_model_path_sup", "../landcover-old/web_tool/data/naip_demo_model.h5",
        # "--in_model_path_ae", "../landcover-old/web_tool/data/naip_autoencoder.h5",
        # "--in_tile_path", "../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf",
        # "--out_model_path_sup", f"{e}/naip_demo_tuned_even.h5",
        # "--out_model_path_ae", f"{e}/naip_ae_tuned_even.h5",
        # "--num_classes", "2",
        # "--gpu", "1",
        # "--exp", e,
        # "--even", "even"])
        # # Imbalanced
        # subprocess.call(["python", "./generate_tuned_model_v2.py",
        # "--in_geo_path", "../notebooks/all_corrections_no_dups.geojson",
        # "--in_model_path_sup", "../landcover-old/web_tool/data/naip_demo_model.h5",
        # "--in_model_path_ae", "../landcover-old/web_tool/data/naip_autoencoder.h5",
        # "--in_tile_path", "../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf",
        # "--out_model_path_sup", f"{e}/naip_demo_tuned_even.h5",
        # "--out_model_path_ae", f"{e}/naip_ae_tuned_even.h5",
        # "--num_classes", "2",
        # "--gpu", "1",
        # "--exp", e,
        # "--even", "uneven"])

    # Batch test inference
    # for e in exps:
    #     temp_files = glob.glob(e + "/*")
    #     for f in temp_files:
    #         mtype = (f.split("/")[2][4:])
    #         mname = (f.split("/")[3][:-3])
    #         e_folder = (e.split("/")[1])
    #         out_fn = ("./" + e_folder + "/test_inference_" + mtype + "/" + mname + ".tif")
    #         print(out_fn)
    #         subprocess.call(["python","./test_inference.py",
    #         "--input_fns", "../../../media/disk2/datasets/maaryland_naip_2017/39077/m_3907719_sw_18_1_20170628.mrf",
    #         "--output_fns", out_fn,
    #         "--model", f,
    #         "--gpu", "1"])

    pass

if __name__ == "__main__":
    main()
