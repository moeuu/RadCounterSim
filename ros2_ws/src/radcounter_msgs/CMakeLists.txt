cmake_minimum_required(VERSION 3.22)
project(radcounter_msgs)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RadiationMeasurement.msg"
  "msg/SourceEstimate.msg"
  "msg/CountermeasureStatus.msg"
  "action/MeasureRadiation.action"
  "action/ExecuteCountermeasure.action"
  "srv/GetDoseMap.srv"
  "srv/EvaluateCountermeasure.srv"
  "srv/ResetEpisode.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
