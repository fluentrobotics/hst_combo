# cmake_minimum_required(VERSION 2.8.3)
# project(cohan_msgs)

# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_COMPILER_ARG1 -std=c++11)

# find_package(catkin REQUIRED COMPONENTS
#   roscpp
#   message_generation
#   geometry_msgs
#   nav_msgs
#   tf
#   visualization_msgs
# )

# # # add message, service and action files
# add_message_files(DIRECTORY msg
#   FILES
#   AgentPath.msg
#   AgentPathArray.msg
#   AgentTimeToGoal.msg
#   AgentTimeToGoalArray.msg
#   AgentTrajectory.msg
#   AgentTrajectoryArray.msg
#   AgentType.msg
#   TrackedSegment.msg
#   TrackedAgent.msg
#   TrackedAgents.msg
#   TrackedSegmentType.msg
#   Trajectory.msg
#   TrajectoryStamped.msg;
#   TrajectoryArray.msg
#   TrajectoryPoint.msg
#   StateArray.msg
#   AgentMarker.msg
#   AgentMarkerStamped.msg
#   PointArray.msg
#   PassageType.msg
# )

# add_service_files(
#   FILES
#   Optimize.srv
#   GetTrajectory.srv
# )

# # # generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   geometry_msgs
#   nav_msgs
#   std_msgs
# )


# # # declare catkin package
# catkin_package(
#   CATKIN_DEPENDS
#   roscpp
#   message_runtime
#   geometry_msgs
#   nav_msgs
#   tf
#   visualization_msgs
# )


cmake_minimum_required(VERSION 3.8)
project(cohan_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AgentPath.msg"
  "msg/AgentPathArray.msg"
  "msg/AgentTimeToGoal.msg"
  "msg/AgentTimeToGoalArray.msg"
  "msg/AgentTrajectory.msg"
  "msg/AgentTrajectoryArray.msg"
  "msg/AgentType.msg"
  "msg/TrackedSegment.msg"
  "msg/TrackedAgent.msg"
  "msg/TrackedAgents.msg"
  "msg/TrackedSegmentType.msg"
  "msg/Trajectory.msg"
  "msg/TrajectoryStamped.msg"
  "msg/TrajectoryArray.msg"
  "msg/TrajectoryPoint.msg"
  "msg/StateArray.msg"
  "msg/AgentMarker.msg"
  "msg/AgentMarkerStamped.msg"
  "msg/PointArray.msg"
  "msg/PassageType.msg"
  "msg/TransformStampedArray.msg"
  "msg/Predictions.msg"
  "msg/AgentPrediction.msg"
  "msg/AgentPredictionArray.msg"
  "srv/Optimize.srv"
  "srv/GetTrajectory.srv"
  DEPENDENCIES geometry_msgs std_msgs nav_msgs# Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)
ament_export_dependencies(rosidl_default_runtime)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

ament_package()
