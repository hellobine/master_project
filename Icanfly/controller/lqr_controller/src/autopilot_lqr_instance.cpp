#include "autopilot/autopilot.h"
#include "lqr_controller/lqr_controller.h"
#include "lqr_controller/lqr_controller_params.h"


int main(int argc, char **argv) {
  
  ros::init(argc, argv, "autopilot_lqr");
  autopilot::AutoPilot<lqr::LQRController,lqr::LQRControllerParams> autopilot;

  ros::spin();
  return 0;
}
