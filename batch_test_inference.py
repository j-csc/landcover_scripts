import subprocess
import glob

def main():
    temp_files = glob.glob("./tmp_ae_uneven/*")
    for m in temp_files:
        out_fn = "./test_inference_ae_uneven/" + m.split("/")[2][:-3] + ".tif"
        subprocess.call(["python","./test_inference.py",
         "--input_fns", "../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf",
         "--output_fns", out_fn,
         "--model", m,
         "--gpu", "1"])
    pass

if __name__ == "__main__":
    main()