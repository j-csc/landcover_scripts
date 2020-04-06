import subprocess
import glob

def main():
    exps = (glob.glob("./new/experiment*"))

    # Generate model
    for e in exps:
        # Balanced
        subprocess.call(["python", "./generate_tuned_model_v2.py",
        "--in_geo_path", "../notebooks/all_corrections_no_dups.geojson",
        "--in_model_path_sup", "../landcover-old/web_tool/data/naip_demo_model.h5",
        "--in_model_path_ae", "../landcover-old/web_tool/data/naip_autoencoder.h5",
        "--in_tile_path", "../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf",
        "--out_model_path_sup", f"./new/{e}/naip_demo_tuned_even.h5",
        "--out_model_path_ae", f"./new/{e}/naip_ae_tuned_even.h5",
        "--num_classes", "2",
        "--gpu", "1",
        "--exp", e,
        "--even", "even"])
        # Imbalanced
        subprocess.call(["python", "./generate_tuned_model_v2.py",
        "--in_geo_path", "../notebooks/all_corrections_no_dups.geojson",
        "--in_model_path_sup", "../landcover-old/web_tool/data/naip_demo_model.h5",
        "--in_model_path_ae", "../landcover-old/web_tool/data/naip_autoencoder.h5",
        "--in_tile_path", "../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf",
        "--out_model_path_sup", f"./new/{e}/naip_demo_tuned_even.h5",
        "--out_model_path_ae", f"./new/{e}/naip_ae_tuned_even.h5",
        "--num_classes", "2",
        "--gpu", "1",
        "--exp", e,
        "--even", "uneven"])

    # Batch test inference
    # for e in exps:
    #     temp_files = glob.glob(e + "/tmp_*/")
    #     print(e)
    #     for f in temp_files:
    #         all_models = glob.glob(f+"*")
    #         for model in all_models:
    #             mtype = (model.split("/")[3][4:])
    #             mname = (model.split("/")[4][:-3])
    #             out_fn = (e + "/test_inference_" + mtype + "/" + mname + ".tif")
    #             subprocess.call(["python","./test_inference.py",
    #             "--input_fns", "../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf",
    #             "--output_fns", out_fn,
    #             "--model", model,
    #             "--gpu", "1"])

    pass

if __name__ == "__main__":
    main()