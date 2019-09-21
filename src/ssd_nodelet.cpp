#include "ssd_ros/ssd_nodelet.h"

PLUGINLIB_EXPORT_CLASS(ObjectDetectionWrapper, nodelet::Nodelet)

ObjectDetectionWrapper::ObjectDetectionWrapper()
{
}

ObjectDetectionWrapper::~ObjectDetectionWrapper()
{
}

void ObjectDetectionWrapper::onInit()
{
  NODELET_DEBUG("Initializing nodelet...");
  inst_.reset(new ObjectDetection(getNodeHandle()));
}
