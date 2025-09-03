# YOLOPv2 ROS Node

This package converts YOLOPv2 to a ROS node to perform lane segmentation and publish the results.

## Features
- Performs lane segmentation using YOLOPv2 model on camera images
- Publishes segmentation results as ROS topics
- Works with the lane_detection_fusion node in creamsooo_fusion package

## Dependencies
- ROS (Tested on: ROS Noetic)
- PyTorch
- OpenCV
- NumPy

## Installation
```bash
# Navigate to the src directory of your catkin workspace
cd ~/catkin_ws/src

# Copy or clone the package
# (Skip if already installed)

# Return to the workspace and build
cd ~/catkin_ws
catkin_make

# Source the workspace setup
source devel/setup.bash
```

## Usage
1. Verify YOLOPv2 model weights: 
   Ensure the weights file is present at `/home/vision/gigacha/src/VISION/lane/YOLOPv2/data/weights/yolopv2.pt`.

2. Check configuration: 
   Adjust parameters in `config/yolopv2_params.yaml` as needed.

3. Run standalone:
   ```bash
   roslaunch yolopv2_ros yolopv2_node.launch
   ```

4. Run with creamsooo_fusion package:
   ```bash
   roslaunch creamsooo_fusion lane_detection_fusion.launch
   ```

## Published Topics
- `/yolopv2/yolo_lane_seg` (sensor_msgs/Image): Lane segmentation result (grayscale image, lanes are white)
- `/yolopv2/yolo_lane_seg_bev` (sensor_msgs/Image): Lane segmentation result in bird's eye view
- `/yolopv2/yolo_drive_area` (sensor_msgs/Image): Drivable area segmentation result
- `/yolopv2/yolo_result_image` (sensor_msgs/Image): Visualized result on the original image

## Subscribed Topics
- `/image_jpeg/compressed` (sensor_msgs/CompressedImage) or `/image_raw` (sensor_msgs/Image): Camera image

## Parameters
- `~weights` (string): Path to YOLOPv2 model weights file
- `~img_size` (int): Input image size (default: 640)
- `~image_topic` (string): Image topic to subscribe to (default: `/image_raw`)
- `~compressed_input` (bool): Whether to use compressed image input (default: false)
- `~device` (string): Device to use for inference (e.g., '0' for CUDA, 'cpu' for CPU)

## Creator
Chaemin Park # CREAM_IONIQ
