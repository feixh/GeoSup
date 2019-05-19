# Geo-Supervised Visual Depth Prediction

[Xiaohan Fei](https://feixh.github.io), [Alex Wong](http://web.cs.ucla.edu/~alexw/), and [Stefano Soatto](http://web.cs.ucla.edu/~soatto/)

[UCLA Vision Lab](http://vision.ucla.edu/)


If you find the [paper][icra19_paper] or the code in this repo useful, please cite the following paper:

```
@incollection{feiWS19,
  author    = {Fei, X. and Wong, A. and Soatto, S.},
  title     = {Geo-Supervised Visual Depth Prediction},
  booktitle = {Proceedings of the International Conference on Robotics and Automation (ICRA)},
  year      = {2019},
  month     = {May}
}
```


Related materials:
- [project site](https://feixh.github.io/projects/icra19/index.html)
- [VISMA](https://github.com/feixh/VISMA) dataset used in the experiments.
- [VISMA2][visma2_data] dataset, a large-scale extension of the original VISMA, built for the development of learning-based visual-inertial sensor fusion.

[icra19_paper]: https://arxiv.org/abs/1807.11130v3.pdf
[icra19_code]: https://github.com/feixh/GeoSup
[visma2_data]: https://www.dropbox.com/s/s9nrx9eoen4tno0/rs_d435i_recording.tar.gz?dl=0

---

## VISMA2 dataset

The VISMA2 dataset contains raw monocular video streams of VGA size (640x480) @ 30 Hz, inertial measurements @ 400 Hz, and depth streams of VGA size @ 30 Hz stored in [rosbags](http://wiki.ros.org/rosbag). 

In addition to the raw data streams, we also provide the camera trajectory estimated by the visual-inertial odometry (VIO) system developed at our lab. The time-stamped camera poses and the camera-to-body alginment will be used to bring gravity from the spatial frame to the camera frame in the experiments. The trajectories are stored in binary [protbufs](https://developers.google.com/protocol-buffers/) defined by custom protocols.

To prepare the training data, you need to 
1. construct image triplets,
2. compute the transformation to bring gravity to the camera frame, and 
3. extract the depth image for validation/testing, and optionally
4. \* compute segmentation masks to apply our regularizer *selectively*
5. \* at training time, if you want to replace the auxiliary pose network with the pose estimated by VIO, you need to compute the relative camera poses from the trajectories.

\* While steps 1 - 3 are needed for training a depth predictor on monocular videos for most methods, steps 4 & 5 marked with an asterisk are specializaed to our proposed training pipeline.

We provide a script to do so, which requires extra dependencies. If you plan to parse the raw data yourself, please check the "Data preparation" section below on how to use the script. 

You can also skip the data preparation step and try out a preprocessed subset of VISMA2 too. See the instructions below on how to train on the preprocessed data.

## Train on preprocessed data

A preprocessed subset of VISMA2 can be found [here](https://www.dropbox.com/s/kccsd0h0wg85ytx/copyrooms.tar.gz?dl=0). Follow the instructions below to use it.

1. Download the tar ball and unzip it into your directory of choice, say `/home/feixh/Data/copyrooms`. And set the environment variable `export EXAMPLEPATH=/home/feixh/Data/copyrooms`. (Note in your data folder, you should see copyroom1, copyroom2, train.txt, etc.)


4. In your terminal, go to `GeoSup/GeoNet` sub-directory, and execute the following command. Note you should replace `example_checkpoints` with directory of your choice to store checkpoints.

```bash
python geonet_main.py \
  --mode train_rigid \
  --dataset_dir $EXAMPLEPATH \
  --checkpoint_dir example_checkpoints \
  --learning_rate 1e-4 \
  --seq_length 3 \
  --batch_size 4 \
  --max_steps 80000 \
  --summary_freq 20 \
  --disp_smooth_weight 1.0 \
  --dispnet_encoder vgg \
  --img_height 240 \
  --img_width 320 \
  --datatype void \
  --validation_dir $EXAMPLEPATH \
  --validation_freq 2000 \
  --use_slam_pose \
  --use_sigl \
  --sigl_loss_weight 0.5
```

For the meaning of each of the arguments, see how they are defined at the top of `geonet_main.py`. Here, we clarify some of the most interesting ones:
- `use_sigl`: If set, impose the *semantically informed geometric loss (SIGL)* to the baseline model.
- `sigl_loss_weight`: The weight for the SIGL loss.
- `disp_smooth_weight`: The weight for the piece-wise smoothness loss.
- `use_slam_pose`: To use pose estimated by the VIO instead of the pose network. This is most useful for data with challenging motion. See the experiment section of our paper for more details.
- `dispnet_encoder`: The architecture of the encoder, can be either `vgg` or `resnet50`.

Note, the code is built on top of the GeoNet model of Yin *et al.* which jointly estimates depth and flow, but we only use it for depth prediction, the flow network is not used and maintained here.

## Data preparation

### Install ROS
To prepare the training data yourself, you need to install the [Robot Operating System (ROS)](http://www.ros.org/) to parse rosbags. Follow instructions on the website to install ROS.

### Download data
Once you have ROS properly installed. Download the VISMA2 dataset from [here][visma2_data], and unzip it into your folder of choice, say, `/home/feixh/Data/VISMA2`. For convenience, let's set the environment variable 

`export VISMA2PATH=/home/feixh/Data/VISMA2`

The VISMA2 folder should contain a list of subfolders, each of which is named after the place where the data is recorded, e.g., classroom0, copyroom1, and stairs3, etc. In each subfolder, there is a `raw.bag` file containing the raw recorded data, and a `dataset` file containing the trajectory and other meta information dumped by our VIO system. There are also two dataset files, namely, `dataset_500` and `dataset_1500`, which are from different runs of the VIO, and can be ignored for now.

### Parse the raw data 

In terminal, first set the environment variable `VISMA2OUTPATH` pointing to the output directory where the parsed dataset should be kept. For instance

`export VISMA2OUTPATH=/home/feixh/Data/visma2_parsed`

then parse the dataset:

```bash
python setup/setup_dataset_visma2.py \
  --recording-dir $VISMA2PATH \
  --output-root  $VISMA2OUTPATH \
  --temporal-interval 5 \
  --spatial-interval 0.01
```

The meaning of the arguments is quite straightforward, and more detailed documentation can be found in the `setup/setup_dataset_visma2.py` script.

After running this script, in the folder `$VISMA2OUTPATH`, you will see five subfolders, namely, `copyroom0~copyroom4`. In each subfolder, you will find the follows:
1. `K.npy` as the camera intrinsics
2. `rgb` folder, which contains a list of image triplets concatenated horizontally. The center image is called the "reference" image. Each file is named after the timestamp of the reference image.
2. `depth` folder, which contains a list of depth images in `.npy` format. Each depth image corresponds to the reference image in the triplet which has the same filename.
3. `pose` folder, which contains the relative camera pose between the other two images in the triplet and the reference, and the rotation to bring gravity to the camera frame of the reference. Each pose file is stored in `.pkl` format, and has the timestamp of the reference image as the filename.

To parse the sequences of your interests, you can add them to the `sequences` varialbe at the top of the `setup_dataset_visma2.py` script.  For now, as you might have noticed, only copyroom0~copyroom4 are added to the variable.

### Prepare the segmentation masks

Once you have parsed the raw data and successfully extracted the image triplets, you can run your favoriate semantic segmentation system to obtain segmentation masks, which will be used to regularize depth predictions selectively in training.

We give an example on how to use PSP-Net to segment the "copyrooms" subset of VISMA2, which we just parsed. We assume the environment variable `$VISMA2OUTPATH` has been set properly in the previous step.


1. First, download the trained model from [here](https://www.dropbox.com/s/l55doruz1dsfvmt/pspnet.tar.gz?dl=0). You can also follow the `README` in `GeoSup/PSPNet` to get the trained model provided by the authors of PSPNet.
2. Unzip the tarball into `GeoSup/PSPNet/model` directory. Make sure the path of your checkpoint relative to your project directory looks like this: `GeoSup/PSPNet/model/ade20k_model/pspnet50/modep.ckpt-0.data` for the model trained on ADE20K indoor dataset, and `GeoSup/PSPNet/model/cityscapes_model/pspnet101/model.ckpt-0.data`. Otherwise, the weights cannot be found with the default setting. You can use a different path, but then you need to specify the path properly when you run the script below.
3. In your terminal, go to `GeoSup/PSPNet`, and execute the following command:

```bash
python inference_visma2.py --dataset ade20k --dataroot $VISMA2OUTPATH
```

- `dataset` argument specifies models trained on which dataset to be used: ade20k for indoors and cityscapes for outdoors
- `dataroot` argument should point to the output root directory of the parsed data 
- Other arguments can be found in the `get_arguments` function of `inference_visma2.py` script.

After executing this command, in each "copyroom" folder, you will find an extra `segmentation` folder along with the `depth`, `pose` and `rgb` folders which you obtained in the last step. The `segmentation` folder contains a list of segmentation masks in `.npy` format named after the timestamp of the image being segmented.

### Split the data

Once you have all the data ready, you can train the model. But before doing that, you need to generate train/test/val splits and saved them as txt files. See the text files (used for the copyroom example) in `GeoSup/GeoNet/data/void` as an template, and the `GeoSup/GeoNet/visma_dataloader.py` script on how the dataloader interacts with the file lists.

## Visualize depth prediction

We also provide the depth prediction of the copyroom subset -- both from the baseline and ours, and a script to show a head-to-head comparison. The prediction of the baseline GeoNet model and GeoNet+SIGL (ours) can be found [here](https://www.dropbox.com/s/wx2m0juwvbx5i5m/VOID_predictions.tar.gz?dl=0). Note, in training both models, we use the pose estimated by the VIO instead of the pose network, i.e., the `use_slam_pose` option is on during training these models.

1. Download the prediction, and unzip the tarball into the directory of your choice, say, `/home/feixh/Data/prediction`. For convenience, point an env variable `$PREDPATH` to it.
2. In `GeoSup/visualization/visualize_visma2.py`, change `project_dir` to your project root directory, and point `validation_dir` to where you keep the parsed dataset. 
3. Now, go to `GeoSup/visualization` and

```bash
python visualize_visma2.py $PREDPATH/GeoNet_depth.npy $PREDPATH/GeoNet_SIGL_depth.npy
```

If everything works properly, you will see a figure showing the input RGB image, the ground-truth depth, the prediction from both the baseline and ours, and the associated error maps.

--- 
## References

[1] GeoNet: https://github.com/yzcjtr/GeoNet

[2] PSPNet: https://github.com/hellochick/PSPNet-tensorflow