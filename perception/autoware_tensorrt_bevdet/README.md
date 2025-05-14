## Prerequestics

- Tensorrt 10.8.0.43
- CUDA 12.4
- cuDNN 8.9.2

## Trained Models

Download the [bevdet_one_lt_d.onnx](https://drive.google.com/file/d/1eMGJfdCVlDPBphBTjMcnIh3wdW7Q7WZB/view?usp=sharing) of trained models in the below path:
   
   ```bash
   $HOME/autoware_data/tensorrt_bevdet
   ```

The `BEVDet` model was trained in `NuScenes` dataset for 20 epochs.

## Test Tensorrt BEVDet Node with Nuscenes

1. Integerate this branch changes in your **autoware_universe/perception** directory

2. Include this [bevdet_vendor pr](https://github.com/autowarefoundation/bevdet_vendor/pull/1) in **src/universe/external/bevdet_vendor** as this supports fp16 precision and api support for Tensorrt 10.x.x

3. To play ros2 bag of nuScenes data
   
   ```bash

   cd autoware/src
   git clone https://github.com/Owen-Liuyuxuan/ros2_dataset_bridge
   cd ..
   
   nano src/ros2_dataset_bridge/launch/nuscenes_launch.xml

   # Modify the below default to point nuscenes dataset!! Also control the publishing frequency of the data stream.

   <arg name="NUSCENES_DIR" default="<nuscenes_dataset_path>"/>
   <arg name="NUSCENES_VER" default="v1.0-trainval"/> 
   
   # Build the autoware

   colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
   
   source install/setup.bash # install/setup.zsh or install/setup.sh for your own need.
   source /opt/ros/humble/setup.bash
   
   # this will launch the data publisher / rviz / GUI controller
   ros2 launch ros2_dataset_bridge nuscenes_launch.xml

   # if no nuscenes boxes visible in rivz, make sure GUI controller "Stop" checkbox is unchecked and click the "OK" tab.
   ```

4. Launch `tensorrt_bevdet_node`

   ```bash
   
   ros2 launch autoware_tensorrt_bevdet tensorrt_bevdet.launch.xml

   # By default precision mode is fp16, to launch with precision mode fp32

   ros2 launch autoware_tensorrt_bevdet tensorrt_bevdet.launch.xml precision:=fp32
   ```
