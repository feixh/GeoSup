## KITTI
1. In `predictions/visualize_kitti.py`, specify the kitti raw data directory via `data_dir`; specify the project root directory via `project_dir`.
2. For Godard, set `mode` to `invdepth`. For GeoNet, set `mode` to `depth`.
3. In `predictions` directory, interpolate the ground truth sparse depth by invoking 
```
python visualize_kitti.py interp
```
4. Once you have the interpolated ground truth, run the visualization script as follows:
```
python visualize_kitti.py \
path/to/your/prediction/of/model/A.npy tag_for_model_A \
path/to/your/prediction/of/model/B.npy tag_for_model_B
```
Note, you need to pair the path to the npy file of the prediction and a tag for that model. For instance
```
python visualize_kitti.py \
predictions/GeoNet_depth.npy GeoNet \
predictions/GeoNet_SIGL_depth.npy GeoNet_SIGL
```

## VISMA2
