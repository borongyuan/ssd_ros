#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "object_msgs/ObjectsInBoxes.h"

class ObjectDetection
{
public:
  ObjectDetection(const ros::NodeHandle& nh);
  ~ObjectDetection();

  struct parameters
  {
    std::string label_filename;
    std::string plan_filename;
    int num_classes;
    float visualize_threshold;

    parameters()
    {
      ros::param::param<std::string>("/ssd_infer/label_filename", label_filename, "/home/jetbot/catkin_ws/src/ssd_ros/"
                                                                                  "data/ssd_coco_labels.txt");
      ros::param::param<std::string>("/ssd_infer/plan_filename", plan_filename,
                                     "/home/jetbot/catkin_ws/src/ssd_ros/data/ssd_mobilenet_v1_coco.plan");
      ros::param::param<int>("/ssd_infer/num_classes", num_classes, 91);
      ros::param::param<float>("/ssd_infer/visualize_threshold", visualize_threshold, 0.5);
    }
  };
  const struct parameters params;

private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber color_sub;
  ros::Publisher bboxes_pub;
  object_msgs::ObjectsInBoxes bboxes_msg;

  void colorCallback(const sensor_msgs::ImageConstPtr& msg);
};
