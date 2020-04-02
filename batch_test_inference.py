import subprocess
import glob

def main():
    exps = (glob.glob("experiment*"))
    for e in exps:
        if e != "experiment1":
            temp_files = glob.glob("./"+e + "/tmp_*/")
            for f in temp_files:
                all_models = glob.glob(f+"*")
                for model in all_models:
                    mtype = (model.split("/")[2][4:])
                    mname = (model.split("/")[3][:-3])
                    out_fn = "./test_inference_" + mtype + mname + ".tif"
                    subprocess.call(["python","./test_inference.py",
                    "--input_fns", "../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf",
                    "--output_fns", out_fn,
                    "--model", model,
                    "--gpu", "1"])

    # Generate model
    # for e in exps:
    #     if (e == "experiment2" or e == "experiment4" or e == "experiment5" or e == "experiment6" or e == "experiment8"or e == "experiment9"):
    #         subprocess.call(["python", "./generate_tuned_model_v2.py",
    #         "--in_geo_path", "../notebooks/all_corrections_no_dups.geojson",
    #         "--in_model_path", "../landcover-old/web_tool/data/naip_demo_model.h5",
    #         "--in_tile_path", "../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf",
    #         "--out_model_path", "./naip_demo_tuned_even.h5",
    #         "--num_classes", "2",
    #         "--gpu", "1",
    #         "--exp", e])
    pass

if __name__ == "__main__":
    main()