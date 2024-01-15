# 2V-CBCT

Code for 2V-CBCT: Two-orthogonal-projection based CBCT reconstruction and dose calculation from real CBCT projection data

1. Prepare your data and check all paths.

2. Run mk_vol_patches.py to prepare the training data for CR-Net.

3. Run train_3d.py to train CR-Net.

4. Run test_3dresunet_and_prepare_data_for_fine_tune.py and mk_2d_slice.py to prepare the training data for FT-Net.

5. Run train_2d.py to train FT-Net. 

6. Run test_2d.py to get the final results.
