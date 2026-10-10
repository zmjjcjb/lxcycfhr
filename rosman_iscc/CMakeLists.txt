cmake_minimum_required(VERSION 3.0.2)
project(rosman_iscc)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  std_srvs
  cv_bridge
  geometry_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    std_srvs
    cv_bridge
    geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/iscc_controller.py
  scripts/line_following_node.py
  scripts/hurdles_node.py
  scripts/climb_stairs_node.py
  scripts/descend_stairs_node.py
  scripts/color_detection_node.py
  scripts/orange_guide_node.py
  scripts/orange_line_follow_node.py
  scripts/sandbag_throw_node.py
  scripts/green_pick_node.py
  scripts/green_return_place_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
