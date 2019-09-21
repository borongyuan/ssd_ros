#include "nodelet/nodelet.h"
#include "pluginlib/class_list_macros.h"
#include "ssd_ros/ssd_node.h"

class ObjectDetectionWrapper : public nodelet::Nodelet

{
public:
  ObjectDetectionWrapper();
  ~ObjectDetectionWrapper();
  virtual void onInit();
  boost::shared_ptr<ObjectDetection> inst_;
};
