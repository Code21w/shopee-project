/**
 * @file hello_moveit_v2.cpp
 * @brief A simple ROS 2 and MoveIt 2 program to control a robot arm (runtime target pose via ROS params)
 */

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.h>                              // RPY -> Quaternion 변환
#include <moveit/move_group_interface/move_group_interface.hpp>     // .hpp 사용

int main(int argc, char * argv[])
{
  // Start up ROS 2
  rclcpp::init(argc, argv);

  // Node 생성: 커맨드라인 파라미터를 자동 선언/적용
  auto const node = std::make_shared<rclcpp::Node>(
    "hello_moveit",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
  );
  auto const logger = rclcpp::get_logger("hello_moveit");

  using moveit::planning_interface::MoveGroupInterface;
  auto arm_group_interface = MoveGroupInterface(node, "arm");

  // 기본 플래닝 파이프라인/플래너 설정
  arm_group_interface.setPlanningPipelineId("ompl");
  arm_group_interface.setPlannerId("RRTConnectkConfigDefault");

  // 플래닝 조건(여유있게)
  arm_group_interface.setPlanningTime(5.0);
  arm_group_interface.setGoalPositionTolerance(0.005);      // 5 mm
  arm_group_interface.setGoalOrientationTolerance(0.05);    // ~3 deg
  arm_group_interface.setMaxVelocityScalingFactor(0.5);
  arm_group_interface.setMaxAccelerationScalingFactor(0.5);
  arm_group_interface.setPoseReferenceFrame("world");
  arm_group_interface.setEndEffectorLink("tcp");
  arm_group_interface.setStartStateToCurrentState();

  RCLCPP_INFO(logger, "Planning pipeline: %s", arm_group_interface.getPlanningPipelineId().c_str());
  RCLCPP_INFO(logger, "Planner ID: %s", arm_group_interface.getPlannerId().c_str());
  RCLCPP_INFO(logger, "Planning time: %.2f", arm_group_interface.getPlanningTime());

  // --- 런타임 파라미터에서 타깃 포즈 읽기 ---
  // 기본값(안전·쉬운 포즈)
  double x = 0.15, y = 0.0, z = 0.20;
  double qx = 0.0, qy = 0.0, qz = 0.0, qw = 1.0;   // Quaternion 기본값
  double rx = 0.0, ry = 0.0, rz = 0.0;             // RPY(라디안) 기본값

  node->get_parameter_or("target_x", x, x);
  node->get_parameter_or("target_y", y, y);
  node->get_parameter_or("target_z", z, z);

  // 사용자가 RPY 중 하나라도 주면 RPY 우선 적용, 아니면 Quaternion 사용
  bool use_rpy = node->has_parameter("target_rx") ||
                 node->has_parameter("target_ry") ||
                 node->has_parameter("target_rz");

  if (use_rpy) {
    node->get_parameter_or("target_rx", rx, rx);
    node->get_parameter_or("target_ry", ry, ry);
    node->get_parameter_or("target_rz", rz, rz);
    tf2::Quaternion q; q.setRPY(rx, ry, rz);
    qx = q.x(); qy = q.y(); qz = q.z(); qw = q.w();
    RCLCPP_INFO(logger, "Using RPY (rad): rx=%.4f ry=%.4f rz=%.4f", rx, ry, rz);
  } else {
    node->get_parameter_or("target_qx", qx, qx);
    node->get_parameter_or("target_qy", qy, qy);
    node->get_parameter_or("target_qz", qz, qz);
    node->get_parameter_or("target_qw", qw, qw);
    RCLCPP_INFO(logger, "Using Quaternion: qx=%.4f qy=%.4f qz=%.4f qw=%.4f", qx, qy, qz, qw);
  }

  geometry_msgs::msg::PoseStamped target_pose;
  target_pose.header.frame_id = "world";
  target_pose.header.stamp = node->now();
  target_pose.pose.position.x = x;
  target_pose.pose.position.y = y;
  target_pose.pose.position.z = z;
  target_pose.pose.orientation.x = qx;
  target_pose.pose.orientation.y = qy;
  target_pose.pose.orientation.z = qz;
  target_pose.pose.orientation.w = qw;

  RCLCPP_INFO(logger, "Target pose (world): xyz=(%.3f, %.3f, %.3f)", x, y, z);

  // 목표 자세 설정
  arm_group_interface.setPoseTarget(target_pose);

  // 계획 생성
  auto const [success, plan] = [&arm_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(arm_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();

  // 실행
  if (success) {
    RCLCPP_INFO(logger, "Planning Succeeded. Executing...");
    arm_group_interface.execute(plan);
  } else {
    RCLCPP_ERROR(logger, "Planning failed!");
  }

  rclcpp::shutdown();
  return 0;
}
