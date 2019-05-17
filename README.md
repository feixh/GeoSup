# Geo-Supervised Visual Depth Prediction

[Xiaohan Fei](https://feixh.github.io), [Alex Wong](http://web.cs.ucla.edu/~alexw/), and [Stefano Soatto](http://web.cs.ucla.edu/~soatto/)

[UCLA Vision Lab](http://vision.ucla.edu/)


If you find the paper or the code in this repo useful, please cite the following paper:

```
Geo-Supervised Visual Depth Prediction
Xiaohan Fei, Alex Wong, and Stefano Soatto
In Proceedings of International Conference on Robotics and Automation (ICRA), 2019.
```

Related materials:
- [project site](https://feixh.github.io/projects/icra19/index.html)
- [paper][icra19_paper], [poster][icra19_poster], and [slides][icra19_slides] as presented at ICRA, 2019.
- [VISMA](https://github.com/feixh/VISMA) dataset used in the experiments.
- [Extended VISMA (eVISMA)]() dataset, a large-scale extension of VISMA built for learning-based visual-inertial sensor fusion.

[icra19_paper]: https://arxiv.org/abs/1807.11130v3.pdf
[icra19_poster]: {{site.url}}/empty.html
[icra19_slides]: {{site.url}}/empty.html
[icra19_code]: https://github.com/feixh/GeoSup

---

## eVISMA dataset

The eVISMA dataset contains raw monocular video streams of VGA size (640x480) @ 30 Hz, inertial measurements @ 400 Hz, and depth streams of VGA size @ 30 Hz stored in [rosbags](http://wiki.ros.org/rosbag). 

In addition to the raw data streams, we also provide the camera trajectory estimated by the visual-inertial odometry (VIO) system developed at our lab. The time-stamped camera poses and the camera-to-body alginment will be used to bring gravity from the spatial frame to the camera frame in the experiments. The trajectories are stored in binary [protbufs](https://developers.google.com/protocol-buffers/) defined by custom protocols.

To prepare the training data, you need to 
- construct image triplets,
- compute the transformation to bring gravity to the camera frame, and 
- extract the depth image for validation/testing.
 
In addition, if you want to replace the auxiliary pose network with the pose estimated by VIO at training time, you need to compute the relative camera poses from the trajectories.

We provide a script to do so, which requires extra dependencies. Check the "Data preparation" section below on how to use the script.

We also provide a subset of eVISMA, which is preprocessed and ready to use. See the section below on how to train on the preprocessed data.

## Train on preprocessed data

The preprocessed subset of eVISMA can be found [here](https://www.dropbox.com/s/kccsd0h0wg85ytx/copyrooms.tar.gz?dl=0).
1. Download the tar ball and unzip it into your directory of choice, say `/local2/Data/VOID`. And set the environment variable `export VOIDPATH=/local2/Data/VOID`. (Note in the VOID folder, you should see copyroom1, copyroom2, train.txt, etc.)


4. In your terminal, go to `GeoSup/GeoNet` sub-directory, and execute the following command. Note you should replace `example_checkpoints` with directory of your choice to store checkpoints.

```
python geonet_main.py \
  --mode train_rigid \
  --dataset_dir $VOIDPATH \
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
  --validation_dir $VOIDPATH \
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

Note, the code is built on top of the GeoNet model of Yin *et al.* which jointly estimates depth and flow, but we only use it for the depth prediction, the flow network is not used maintained here.

## Data preparation

To prepare the training data, you need to install the [Robot Operating System (ROS)](http://www.ros.org/) first. And do the following:

1. Download the raw data in rosbags
2. Download the trajectories in protobufs
3. Run the data preparation script


## Visualize predicted depth


