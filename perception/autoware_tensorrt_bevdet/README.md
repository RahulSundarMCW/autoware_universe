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
   
   #Open the launch file to configure dataset settings:
   
   nano src/ros2_dataset_bridge/launch/nuscenes_launch.xml
   
   # Update the following lines with the correct NuScenes dataset path and set the publishing frequency to 10 Hz for optimal data streaming:
   
   <arg name="NUSCENES_DIR" default="<nuscenes_dataset_path>"/>
   <arg name="NUSCENES_VER" default="v1.0-trainval"/> 
   <arg name="UPDATE_FREQUENCY" default="10.0"/>
   
   # Build Autoware
   
   colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
   
   # Source the environment
   
   source install/setup.bash # install/setup.zsh or install/setup.sh for your own need.
   source /opt/ros/humble/setup.bash
   
   # Launch the data publisher, RViz, and GUI controller:
   
   ros2 launch ros2_dataset_bridge nuscenes_launch.xml
   
   # Tip: If NuScenes boxes are not visible in RViz, ensure the "Stop" checkbox in the GUI controller is unchecked, then click "OK".
   
   # Note: ROS bag playback is limited to 10 Hz, which constrains the BEVDet node to the same rate. However, based on callback execution time, BEVDet can run at up to 35 FPS with FP16 and 17 FPS with FP32.
   ```

4. Launch `tensorrt_bevdet_node`

   ```bash
   
   ros2 launch autoware_tensorrt_bevdet tensorrt_bevdet.launch.xml

   # By default precision mode is fp16, to launch with precision mode fp32

   ros2 launch autoware_tensorrt_bevdet tensorrt_bevdet.launch.xml precision:=fp32
   ```
