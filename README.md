# ssd_ros
SSD ROS node(nodelet) accelerated with TensorRT

Tested on Jetson Nano with TensorRT 5.0.6

## Dependencies
1. [object_msgs](https://github.com/intel/object_msgs)
2. [ros_object_detection_viewer](https://github.com/borongyuan/ros_object_detection_viewer)

## Steps
1. Read TensorRT developer guide and try to run sampleUffSSD in TensorRT samples. We will prepare our network in a similar way with some modifications.
2. Download pretrained [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz). You could try other SSD models from [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), but may be slower than Mobilenet V1.
3. Generate UFF file following README.txt in sampleUffSSD using config.py in this repository.
4. Replace sampleUffSSD.cpp in TensorRT samples with the modified file in this repository.
5. Recompile TensorRT samples and run sampleUffSSD. Now you should get a generated engine file sample_ssd.plan in bin folder.
6. Clone this package and its dependencies to your ROS workspace and make them. Copy sample_ssd.plan to ssd_ros/data and modify the path in ssd_ros/cfg/ssd_mobilenet_v1_coco.yaml.
7. Launch ssd_infer.launch to run as a standalone node. You could also run ssd_ros as nodelet.
8. Launch ssd_viewer.launch to view result.