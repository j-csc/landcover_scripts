import subprocess
import glob
import sys, os, time
from multiprocessing import Process, Queue
import rasterio
 
def do_work(work, gpu_idx):
    while not work.empty():
        fn = work.get()
        filename = (fn.split("/")[-1].split(".")[0])
        # out_fn = f"./post_processed_md_inference/{filename}_processed.geojson"
        # subprocess.call(["python","./test_inference.py",
        #     "--input_fns", fn,
        #     "--output_fns", out_fn,
        #     "--model", "./new/experiment10/tmp_ae_even/ae_tuned_model_even_08_0.17.h5",
        #     "--gpu", str(gpu_idx)])
        # subprocess.call(["python","./post_processing.py",
        #     "--input_fn", fn,
        #     "--output_fn", out_fn,
        #     "--gpu", str(gpu_idx)])s
        p_fn = f"./post_processed_de_inference/{filename}_processed.geojson"
        out_fn = f"./post_processed_de_tif/{filename}_processed.tif"
        f = rasterio.open(fn,"r")
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
            p_fn,
            out_fn
        ]
        subprocess.call(command)
        
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
    # fn_folders = glob.glob("../../../media/disk1/datasets/delaware_data/de_100cm_2017/*")
    # all_fns = []
    # for fn_folder in fn_folders:
    #     fns = (glob.glob(fn_folder + "/*.tif"))
    #     for fn in fns:
    #         all_fns.append(fn)
    # batch_run(all_fns)

    # Batch run post-processing
    fn_folders = glob.glob("./all_test_inference_de/*")
    batch_run(fn_folders)

    # Rasterize


    # Small batch inference
    # exps = (glob.glob("./temp_123/tmp_*"))
    # subprocess.call(["python","./test_inference.py",
    #             "--input_fns", "../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf",
    #             "--output_fns", "./m_3807537_ne_18_1_20170611_sup_even_best.tif",
    #             "--model", "./new/experiment9/tmp_sup_even/sup_tuned_model_even_09_0.11.h5",
    #             "--gpu", "1"])
    # subprocess.call(["python","./test_inference.py",
    #             "--input_fns", "../../../media/disk2/datasets/maaryland_naip_2017/38075/m_3807536_se_18_1_20170611.mrf",
    #             "--output_fns", "./m_3807536_se_18_1_20170611_sup_uneven_best.tif",
    #             "--model", "./new/experiment9/tmp_sup_uneven/sup_tuned_model_uneven_03_0.24.h5",
    #             "--gpu", "1"])
    # subprocess.call(["python","./test_inference.py",
    #             "--input_fns", "../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf",
    #             "--output_fns", "./m_3807536_se_18_1_20170611_ae_even_best.tif",
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
