#include "ssd_ros/ssd_node.h"

extern void setup(std::string labelFilename, std::string planFilename, int numClasses, float th);
extern void destroy(void);
extern object_msgs::ObjectsInBoxes infer(const sensor_msgs::ImageConstPtr& color_msg);

ObjectDetection::ObjectDetection(const ros::NodeHandle& nh) : nh_(nh), it_(nh), params()
{
  setup(params.label_filename, params.plan_filename, params.num_classes, params.visualize_threshold);
  color_sub = it_.subscribe("input_topic", 10, &ObjectDetection::colorCallback, this);
  bboxes_pub = nh_.advertise<object_msgs::ObjectsInBoxes>("output_topic", 10);
}

ObjectDetection::~ObjectDetection()
{
  destroy();
}

void ObjectDetection::colorCallback(const sensor_msgs::ImageConstPtr& color_msg)
{
  bboxes_msg = infer(color_msg);
  bboxes_pub.publish(bboxes_msg);
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "ssd_infer");

  ros::NodeHandle nh("~");
  ObjectDetection ssd(nh);

  ros::spin();

  return 0;
}