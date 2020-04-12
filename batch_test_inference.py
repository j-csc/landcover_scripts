import subprocess
import glob

def main():
    exps = (glob.glob("./new/experiment*"))
    subprocess.call(["python","./test_inference.py",
                "--input_fns", "../../../media/disk2/datasets/maaryland_naip_2017/38075/m_3807536_se_18_1_20170611.mrf",
                "--output_fns", "./m_3807536_se_18_1_20170611_sup_even_best.tif",
                "--model", "./new/experiment9/tmp_sup_even/sup_tuned_model_even_09_0.11.h5",
                "--gpu", "1"])
    subprocess.call(["python","./test_inference.py",
                "--input_fns", "../../../media/disk2/datasets/maaryland_naip_2017/38075/m_3807536_se_18_1_20170611.mrf",
                "--output_fns", "./m_3807536_se_18_1_20170611_sup_uneven_best.tif",
                "--model", "./new/experiment9/tmp_sup_uneven/sup_tuned_model_uneven_03_0.24.h5",
                "--gpu", "1"])
    subprocess.call(["python","./test_inference.py",
                "--input_fns", "../../../media/disk2/datasets/maaryland_naip_2017/38075/m_3807536_se_18_1_20170611.mrf",
                "--output_fns", "./m_3807536_se_18_1_20170611_ae_even_best.tif",
                "--model", "./new/experiment10/tmp_ae_even/ae_tuned_model_even_08_0.17.h5",
                "--gpu", "1"])

    subprocess.call(["python","./test_inference.py",
                "--input_fns", "../../../media/disk2/datasets/maaryland_naip_2017/38075/m_3807536_se_18_1_20170611.mrf",
                "--output_fns", "./m_3807536_se_18_1_20170611_ae_uneven_best.tif",
                "--model", "./new/experiment3/tmp_ae_uneven/ae_tuned_model_uneven_03_0.20.h5",
                "--gpu", "1"])

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
    #     temp_files = glob.glob(e + "/tmp_*/")
    #     if (e.split("/")[-1] == "experiment9"):
    #         for f in temp_files:
    #             all_models = glob.glob(f+"*")
    #             print(len(all_models))
    #             for model in all_models:
    #                 mtype = (model.split("/")[3][4:])
    #                 mname = (model.split("/")[4][:-3])
    #                 out_fn = (e + "/test_inference_" + mtype + "/" + mname + ".tif")
    #                 print(out_fn)
    #                 subprocess.call(["python","./test_inference.py",
    #                 "--input_fns", "../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf",
    #                 "--output_fns", out_fn,
    #                 "--model", model,
    #                 "--gpu", "1"])

    pass

if __name__ == "__main__":
    main()
