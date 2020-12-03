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
        # Test Inference
        #################

        out_fn = f"./test_run/single_tile_4000_inf/{filename}_multi_inference.tif"
        subprocess.call(["python","./test_inference.py",
            "--input_fns", fn,
            "--output_fns", out_fn,
            "--model", "./test_run/single_tile_4000s/ae_tuned_model_04_0.03.h5",
            "--gpu", str(gpu_idx)])
        
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
    
    fn_folders = glob.glob("../../../media/disk2/datasets/all_maryalnd_naip/m_38075*.mrf") # MD
    
    all_fns = []
    for fn_folder in fn_folders:
        fns = (glob.glob(fn_folder + "/*.mrf"))
        print(fns)
        for fn in fns:
            all_fns.append(fn)
    print(fn_folders)


    batch_run(fn_folders)

    pass

if __name__ == "__main__":
    main()
