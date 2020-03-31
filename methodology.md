## Experimentation procedure

1. Obtained 1922 correction labels. 
    - Among which 180 were duplicates.
    - Resulting with **1742** correction labels.
    - Labels were distributed as follows:
        1. Chicken house - 576
        2. Built - 387
        3. Field - 338
        4. Tree Canopy - 275
        5. Water - 166

2. Generate tuned model
    - Command ran: 
    
    ```python generate_tuned_model_v2.py --in_geo_path ../notebooks/all_corrections_no_dups.geojson --in_model_path ../landcover-old/web_tool/data/naip_autoencoder.h5 --in_tile_path ../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf --out_model_path ./naip_autoencoder_tuned_uneven.h5 --num_classes 2 --gpu 1```

    ```python generate_tuned_model_v2.py --in_geo_path ../notebooks/all_corrections_no_dups.geojson --in_model_path ../landcover-old/web_tool/data/naip_autoencoder.h5 --in_tile_path ../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf --out_model_path ./naip_autoencoder_tuned_even.h5 --num_classes 2 --gpu 1```

    ``` python generate_tuned_model_v2.py --in_geo_path ../notebooks/all_corrections_no_dups.geojson --in_model_path ../landcover-old/web_tool/data/naip_demo_model.h5 --in_tile_path ../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf --out_model_path ./naip_demo_tuned_uneven.h5 --num_classes 2 --gpu 1```

   ``` python generate_tuned_model_v2.py --in_geo_path ../notebooks/all_corrections_no_dups.geojson --in_model_path ../landcover-old/web_tool/data/naip_demo_model.h5 --in_tile_path ../landcover-old/web_tool/tiles/m_3807537_ne_18_1_20170611.mrf --out_model_path ./naip_demo_tuned_even.h5 --num_classes 2 --gpu 1```

3. Sampling
    - For even sampling:
        150 samples each -> 750 total
        
    - For uneven sampling:
        375 Chicken houses, 375 random samples from the other classes.

    Example distribution of uneven sampling:

    4.0    375
    3.0    129
    2.0    123
    1.0     73
    0.0     50

    Example distribution of even sampling:

    4.0    150
    3.0    150
    2.0    150
    1.0    150
    0.0    150