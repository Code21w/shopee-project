/**
 * @file plan_around_objects.cpp
 * @brief Demonstrates basic usage of MoveIt with ROS 2 and collision avoidance.
 *
 * This program initializes a ROS 2 node, sets up a MoveIt interface for a robot manipulator,
 * creates collision objects in the scene (a 4-side fence), plans a motion to a target pose
 * while avoiding the collision objects, and executes the planned motion.
 */

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <vector>

int main(int argc, char* argv[])
{
  // Start up ROS 2
  rclcpp::init(argc, argv);

  auto const node = std::make_shared<rclcpp::Node>(
      "plan_around_objects",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  auto const logger = rclcpp::get_logger("plan_around_objects");

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  using moveit::planning_interface::MoveGroupInterface;
  auto arm_group_interface = MoveGroupInterface(node, "arm");

  arm_group_interface.setPlanningPipelineId("ompl");
  arm_group_interface.setPlannerId("RRTConnectkConfigDefault");
  arm_group_interface.setPlanningTime(1.0);
  arm_group_interface.setMaxVelocityScalingFactor(1.0);
  arm_group_interface.setMaxAccelerationScalingFactor(1.0);

  RCLCPP_INFO(logger, "Planning pipeline: %s", arm_group_interface.getPlanningPipelineId().c_str());
  RCLCPP_INFO(logger, "Planner ID: %s", arm_group_interface.getPlannerId().c_str());
  RCLCPP_INFO(logger, "Planning time: %.2f", arm_group_interface.getPlanningTime());

  auto moveit_visual_tools =
      moveit_visual_tools::MoveItVisualTools{node,
                                             "base_link",
                                             rviz_visual_tools::RVIZ_MARKER_TOPIC,
                                             arm_group_interface.getRobotModel()};
  moveit_visual_tools.deleteAllMarkers();
  moveit_visual_tools.loadRemoteControl();

  auto const draw_title = [&moveit_visual_tools](auto text) {
    auto const text_pose = [] {
      Eigen::Isometry3d msg = Eigen::Isometry3d::Identity();
      msg.translation().z() = 1.0;
      return msg;
    }();
    moveit_visual_tools.publishText(text_pose, text, rviz_visual_tools::WHITE,
                                    rviz_visual_tools::XLARGE);
  };

  auto const prompt = [&moveit_visual_tools](auto text) { moveit_visual_tools.prompt(text); };

  auto const draw_trajectory_tool_path =
      [&moveit_visual_tools, jmg = arm_group_interface.getRobotModel()->getJointModelGroup("arm")](
          auto const trajectory) { moveit_visual_tools.publishTrajectoryLine(trajectory, jmg); };

  // Target pose
  auto const arm_target_pose = [&node] {
    geometry_msgs::msg::PoseStamped msg;
    msg.header.frame_id = "base_link";
    msg.header.stamp = node->now();
    msg.pose.position.x = 0.128;
    msg.pose.position.y = -0.266;
    msg.pose.position.z = 0.111;
    msg.pose.orientation.x = 0.635;
    msg.pose.orientation.y = -0.268;
    msg.pose.orientation.z = 0.694;
    msg.pose.orientation.w = 0.206;
    return msg;
  }();
  arm_group_interface.setPoseTarget(arm_target_pose);

  // =========================================================
  // FENCE: world 기준 4면 (좌하단 모서리 = (0,0,0), 바닥에서 0.05 m 띄움)
  // 외곽 크기: X 0.35, Y 0.25 (요청한 35cm x 25cm), 두께 0.005, 높이 0.08
  // 두께는 +X, +Y 방향으로만 확장되도록 코너 앵커 배치
  // =========================================================
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  auto make_box = [&](const std::string& id,
                      double size_x, double size_y, double size_z,
                      double px, double py, double pz) {
    moveit_msgs::msg::CollisionObject obj;
    obj.id = id;
    obj.header.frame_id = "world";  // 월드 좌표 기준
    obj.header.stamp = node->now();

    shape_msgs::msg::SolidPrimitive prim;
    prim.type = shape_msgs::msg::SolidPrimitive::BOX;
    prim.dimensions.resize(3);
    prim.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] = size_x;
    prim.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] = size_y;
    prim.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] = size_z;

    geometry_msgs::msg::Pose pose;
    pose.position.x = px;
    pose.position.y = py;
    pose.position.z = pz;  // 하단 gap + 높이/2
    pose.orientation.x = 0.0;
    pose.orientation.y = 0.0;
    pose.orientation.z = 0.0;
    pose.orientation.w = 1.0;

    obj.primitives.push_back(prim);
    obj.primitive_poses.push_back(pose);
    obj.operation = obj.ADD;

    RCLCPP_INFO(logger,
                "[Fence] %s  size(%.2f, %.2f, %.2f)  pos(%.2f, %.2f, %.2f)",
                id.c_str(), size_x, size_y, size_z, px, py, pz);
    return obj;
  };

  const double span_x = 0.35;   // 전체 X (m)
  const double span_y = 0.25;   // 전체 Y (m)
  const double thick  = 0.005;  // 두께 (m)
  const double height = 0.08;   // 높이
  const double gap    = 0.02;   // 바닥에서 하단 띄움
  const double zc     = gap + height * 0.5;  // 중심 z

  // 남/북(가로): 길이X=span_x, 두께Y=thick
  auto fence_south = make_box("fence_south", span_x, thick, height,
                              /*px=*/span_x * 0.5,
                              /*py=*/thick * 0.5,
                              /*pz=*/zc);

  auto fence_north = make_box("fence_north", span_x, thick, height,
                              /*px=*/span_x * 0.5,
                              /*py=*/span_y - thick * 0.5,
                              /*pz=*/zc);

  // 서/동(세로): 두께X=thick, 길이Y=span_y
  auto fence_west  = make_box("fence_west", thick, span_y, height,
                              /*px=*/thick * 0.5,
                              /*py=*/span_y * 0.5,
                              /*pz=*/zc);

  auto fence_east  = make_box("fence_east", thick, span_y, height,
                              /*px=*/span_x - thick * 0.5,
                              /*py=*/span_y * 0.5,
                              /*pz=*/zc);

  // Mirror fence across X-axis: 15mm narrower in X (span_x) and 65mm tall (gap identical)
  const double span_x_mirror = span_x - 0.015;  // 15 mm narrower width
  const double span_x_center = span_x * 0.5;
  const double half_span_x_mirror = span_x_mirror * 0.5;
  const double span_y_mirror = span_y;          // same depth as original
  const double height_mirror = 0.065;           // 65 mm tall (20 mm shorter than before)
  const double zc_mirror = gap + height_mirror * 0.5;

  auto fence_south_mirror = make_box("fence_south_mirror", span_x_mirror, thick, height_mirror,
                                     /*px=*/span_x_center,
                                     /*py=*/-thick * 0.5,
                                     /*pz=*/zc_mirror);

  auto fence_north_mirror = make_box("fence_north_mirror", span_x_mirror, thick, height_mirror,
                                     /*px=*/span_x_center,
                                     /*py=*/-(span_y_mirror - thick * 0.5),
                                     /*pz=*/zc_mirror);

  auto fence_west_mirror  = make_box("fence_west_mirror", thick, span_y_mirror, height_mirror,
                                     /*px=*/span_x_center - half_span_x_mirror + thick * 0.5,
                                     /*py=*/-span_y_mirror * 0.5,
                                     /*pz=*/zc_mirror);

  auto fence_east_mirror  = make_box("fence_east_mirror", thick, span_y_mirror, height_mirror,
                                     /*px=*/span_x_center + half_span_x_mirror - thick * 0.5,
                                     /*py=*/-span_y_mirror * 0.5,
                                     /*pz=*/zc_mirror);

  planning_scene_interface.applyCollisionObjects(
      std::vector<moveit_msgs::msg::CollisionObject>{
          fence_south, fence_north, fence_west, fence_east,
          fence_south_mirror, fence_north_mirror, fence_west_mirror, fence_east_mirror});
  // =========================================================

  prompt("Press 'next' in the RvizVisualToolsGui window to plan");
  draw_title("Planning");
  moveit_visual_tools.trigger();

  auto const [success, plan] = [&arm_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(arm_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();

  if (success) {
    draw_trajectory_tool_path(plan.trajectory);
    moveit_visual_tools.trigger();
    prompt("Press 'next' in the RvizVisualToolsGui window to execute");
    draw_title("Executing");
    moveit_visual_tools.trigger();
    arm_group_interface.execute(plan);
  } else {
    draw_title("Planning Failed!");
    moveit_visual_tools.trigger();
    RCLCPP_ERROR(logger, "Planning failed!");
  }

  rclcpp::shutdown();
  spinner.join();
  return 0;
}
