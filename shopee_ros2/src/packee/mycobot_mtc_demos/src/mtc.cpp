/**
 * @file mtc.cpp
 * @brief Packee-oriented MTC service node that delegates product verification to the vision stack and
 *        executes MoveIt Task Constructor plans once vision grants control.
 *
 * The node exposes a PackeeMainStartMTC service. Each incoming request contains a list of sequences (product_id + pose data).
 * For every sequence we:
 *   1. Forward the product id to the Packee vision detection service.
 *   2. While vision owns control, continuously monitor the robot joint state without issuing any command.
 *   3. Once vision responds (bool + message), regain control and continue with the next sequence.
 *
 * The node interleaves non-motion hand-offs (vision verification, joint monitoring) with MTC motion plans:
 *   - Each sequence begins from the configured home pose.
 *   - The product id is sent to the vision stack and the robot state is only monitored while vision owns control.
 *   - After a positive vision response, the robot moves to the requested pose, opens the gripper, and returns home.
 * This containerised pattern repeats for every sequence in the service request.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rcpputils/scope_exit.hpp>

#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/task_constructor/container.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers/joint_interpolation.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/storage.h>
#include <moveit/robot_trajectory/robot_trajectory.h>

#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "shopee_interfaces/msg/sequence.hpp"
#include "shopee_interfaces/srv/arm_pick_product.hpp"
#include "shopee_interfaces/srv/packee_arm_packing_complete.hpp"
#include "shopee_interfaces/srv/packee_main_start_mtc.hpp"

namespace mtc = moveit::task_constructor;

namespace {

using SequenceMsg = shopee_interfaces::msg::Sequence;
using ArmPickProduct = shopee_interfaces::srv::ArmPickProduct;
using PackeeMainStartMTC = shopee_interfaces::srv::PackeeMainStartMTC;
using PackingComplete = shopee_interfaces::srv::PackeeArmPackingComplete;

struct SequenceResult {
  SequenceMsg sequence;
  bool success{ false };
  std::string message;
  std::int32_t total_detected{ 0 };
};

geometry_msgs::msg::PoseStamped toPoseStamped(const SequenceMsg& sequence, const std::string& frame_id) {
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = frame_id;
  pose.pose.position.x = sequence.x;
  pose.pose.position.y = sequence.y;
  pose.pose.position.z = sequence.z;

  tf2::Quaternion orientation;
  orientation.setRPY(sequence.rx, sequence.ry, sequence.rz);
  pose.pose.orientation = tf2::toMsg(orientation);
  return pose;
}

}  // namespace

class MtcOrchestrator : public rclcpp::Node {
public:

  explicit MtcOrchestrator(const rclcpp::NodeOptions& options)
  : rclcpp::Node("packee_mtc_orchestrator", options) {
    const auto declare_string = [this](const std::string& name, const std::string& default_value,
                                       const std::string& description) {
      rcl_interfaces::msg::ParameterDescriptor descriptor;
      descriptor.description = description;
      return this->declare_parameter<std::string>(name, default_value, descriptor);
    };

    const auto declare_double = [this](const std::string& name, double default_value,
                                       const std::string& description) {
      rcl_interfaces::msg::ParameterDescriptor descriptor;
      descriptor.description = description;
      return this->declare_parameter<double>(name, default_value, descriptor);
    };
    // 연결
    arm_group_name_ = declare_string("arm_group_name", "arm", "MoveIt planning group for the arm.");
    gripper_group_name_ = declare_string("gripper_group_name", "gripper", "MoveIt group controlling the gripper.");
    gripper_frame_ = declare_string("gripper_frame", "tcp", "Frame used for end-effector IK.");
    gripper_open_pose_ = declare_string("gripper_open_pose", "open", "Named pose representing an open gripper.");
    arm_home_pose_ = declare_string("arm_home_pose", "home", "Named pose representing the home configuration.");
    world_frame_ = declare_string("world_frame", "world", "Reference frame for sequence poses.");

    vision_service_name_ = declare_string(
      "vision_service_name", "/packee/picking/ibvs",
      "Service used to delegate product verification to the IBVS picking interface.");
    packing_complete_service_name_ = declare_string(
      "packing_complete_service_name", "/packee/mtc/finish",
      "Service used to notify Packee MTC completion.");
    joint_state_topic_ = declare_string(
      "joint_state_topic", "joint_states",
      "Topic used for monitoring robot joint state.");

    service_wait_timeout_ms_ = static_cast<std::int64_t>(
      declare_double("service_wait_timeout_ms", 3000.0,
                     "Maximum time (ms) to wait for vision responses."));
    monitor_interval_ms_ = static_cast<std::int64_t>(
      declare_double("monitor_interval_ms", 500.0,
                     "Monitoring timer interval (ms) while vision has control."));
    state_stale_timeout_ms_ = static_cast<std::int64_t>(
      declare_double("state_stale_timeout_ms", 2000.0,
                     "Allowed age (ms) of the last joint state before warning."));
    {
      rcl_interfaces::msg::ParameterDescriptor descriptor;
      descriptor.description = "When true, publish MTC solutions to RViz (may trigger auto execute).";
      publish_solutions_ = this->declare_parameter<bool>("publish_solutions", false, descriptor);
    }

    arm_velocity_scaling_ = declare_double(
      "arm_velocity_scaling_factor", 0.5,
      "Velocity scaling factor applied to arm motions (0.0, 1.0].");
    arm_acceleration_scaling_ = declare_double(
      "arm_acceleration_scaling_factor", 0.5,
      "Acceleration scaling factor applied to arm motions (0.0, 1.0].");
    gripper_velocity_scaling_ = declare_double(
      "gripper_velocity_scaling_factor", 0.5,
      "Velocity scaling factor applied to gripper motions (0.0, 1.0].");
    gripper_acceleration_scaling_ = declare_double(
      "gripper_acceleration_scaling_factor", 0.5,
      "Acceleration scaling factor applied to gripper motions (0.0, 1.0].");

    const auto declare_pause_ms = [this](const std::string& name, double default_value,
                                         const std::string& description) {
      rcl_interfaces::msg::ParameterDescriptor descriptor;
      descriptor.description = description;
      const double value_ms = this->declare_parameter<double>(name, default_value, descriptor);
      return std::chrono::milliseconds(static_cast<std::int64_t>(value_ms));
    };

    pause_before_place_move_ = declare_pause_ms(
      "pause_before_place_move_ms", 1500.0,
      "Delay (ms) before commanding the approach motion toward the place pose.");
    pause_before_descent_ = declare_pause_ms(
      "pause_before_descent_ms", 1500.0,
      "Delay (ms) before descending onto the place pose.");
    pause_before_release_ = declare_pause_ms(
      "pause_before_release_ms", 1500.0,
      "Delay (ms) before opening the gripper to release the object.");
    pause_before_retreat_ = declare_pause_ms(
      "pause_before_retreat_ms", 1500.0,
      "Delay (ms) before retreating upward after releasing the object.");
    pause_before_home_ = declare_pause_ms(
      "pause_before_home_ms", 1500.0,
      "Delay (ms) before returning to the configured home posture.");
    pause_between_sequences_ = declare_pause_ms(
      "pause_between_sequences_ms", 3000.0,
      "Delay (ms) between processing consecutive sequences.");

    start_mtc_service_ = this->create_service<PackeeMainStartMTC>(
      "/packee/mtc/startmtc",
      std::bind(&MtcOrchestrator::handleStartMtc, this, std::placeholders::_1, std::placeholders::_2));

    rclcpp::NodeOptions vision_options;
    vision_options.context(this->get_node_base_interface()->get_context());
    vision_options.use_intra_process_comms(false);
    vision_proxy_node_ = std::make_shared<rclcpp::Node>(
      std::string(this->get_name()) + "_vision_proxy", vision_options);

    vision_client_ = vision_proxy_node_->create_client<ArmPickProduct>(vision_service_name_);
    packing_complete_client_ = vision_proxy_node_->create_client<PackingComplete>(packing_complete_service_name_);

    joint_state_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic_, rclcpp::SensorDataQoS(),
      std::bind(&MtcOrchestrator::handleJointState, this, std::placeholders::_1));

    monitor_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(monitor_interval_ms_),
      std::bind(&MtcOrchestrator::monitorRobotState, this));

    RCLCPP_INFO(this->get_logger(), "Packee MTC orchestrator initialised. Awaiting PackeeMainStartMTC requests.");
  }

private:
  void handleStartMtc(const std::shared_ptr<PackeeMainStartMTC::Request> request,
                      std::shared_ptr<PackeeMainStartMTC::Response> response) {
    {
      std::lock_guard<std::mutex> guard(task_mutex_);
      if (task_in_progress_) {
        response->success = false;
        response->message = "A PackeeMainStartMTC sequence is already in progress.";
        RCLCPP_WARN(this->get_logger(), "Rejected PackeeMainStartMTC request: task already running.");
        return;
      }
      task_in_progress_ = true;
    }

    if (request->sequences.empty()) {
      {
        std::lock_guard<std::mutex> guard(task_mutex_);
        task_in_progress_ = false;
      }
      response->success = false;
      response->message = "No sequences provided.";
      RCLCPP_WARN(this->get_logger(), "PackeeMainStartMTC request rejected: empty sequence list.");
      return;
    }

    ensureMoveGroupInterfaces();

    auto request_copy = *request;
    const std::size_t total_sequences = request_copy.sequences.size();

    response->success = true;
    response->message = "PackeeMainStartMTC request accepted (" + std::to_string(total_sequences) + " sequences).";

    RCLCPP_INFO(this->get_logger(),
                "Accepted PackeeMainStartMTC request: robot_id=%d order_id=%d total_sequences=%zu",
                request_copy.robot_id, request_copy.order_id, total_sequences);

    if (worker_future_.valid()) {
      worker_future_.wait();
    }

    auto self = std::static_pointer_cast<MtcOrchestrator>(shared_from_this());
    worker_future_ = std::async(std::launch::async, [self, request_copy]() mutable {
      self->processSequences(std::move(request_copy));
    });
  }

  void processSequences(PackeeMainStartMTC::Request request) {
    ensureMoveGroupInterfaces();

    auto sequences = request.sequences;
    std::sort(sequences.begin(), sequences.end(),
              [](const SequenceMsg& lhs, const SequenceMsg& rhs) { return lhs.seq < rhs.seq; });

    RCLCPP_INFO(this->get_logger(),
                "Processing %zu sequences for robot_id=%d order_id=%d",
                sequences.size(), request.robot_id, request.order_id);

    std::vector<SequenceResult> results;
    results.reserve(sequences.size());

    bool overall_success = true;
    std::string final_message;

    try {
      for (std::size_t index = 0; index < sequences.size(); ++index) {
        const auto& sequence = sequences[index];
        SequenceResult result;
        result.sequence = sequence;

        std::string vision_message;
        std::int32_t total_detected = 0;

        if (!delegateToVision(request, sequence, vision_message, total_detected)) {
          result.success = false;
          result.message = vision_message;
          result.total_detected = total_detected;
          results.push_back(result);

          final_message = composeFailureMessage(result);
          overall_success = false;
          pauseAfterSequence(sequence.seq);
          break;
        }

        RCLCPP_INFO(this->get_logger(),
                    "Sequence %d (product %d) vision verification succeeded: %s (detected=%d)",
                    sequence.seq, sequence.id, vision_message.c_str(), total_detected);

        std::string execution_message;
        if (!planAndExecuteSequence(sequence, execution_message)) {
          result.success = false;
          result.message = execution_message;
          result.total_detected = total_detected;
          results.push_back(result);

          final_message = composeFailureMessage(result);
          overall_success = false;
          pauseAfterSequence(sequence.seq);
          break;
        }

        RCLCPP_INFO(this->get_logger(),
                    "Sequence %d (product %d) execution complete: %s",
                    sequence.seq, sequence.id, execution_message.c_str());

        result.success = true;
        result.message = "Vision: " + vision_message + " | Exec: " + execution_message;
        result.total_detected = total_detected;
        results.push_back(result);
        pauseAfterSequence(sequence.seq);
      }
    } catch (const std::exception& e) {
      overall_success = false;
      final_message = std::string("Unhandled exception while processing sequences: ") + e.what();
      RCLCPP_ERROR(this->get_logger(), "%s", final_message.c_str());
    }

    if (overall_success) {
      final_message = composeSuccessMessage(results);
      RCLCPP_INFO(this->get_logger(),
                  "Completed all sequences for robot_id=%d order_id=%d: %s",
                  request.robot_id, request.order_id, final_message.c_str());
    } else if (final_message.empty()) {
      final_message = "Sequence processing aborted.";
      RCLCPP_ERROR(this->get_logger(), "%s", final_message.c_str());
    } else {
      RCLCPP_ERROR(this->get_logger(), "%s", final_message.c_str());
    }

    if (!sendPackingComplete(request, overall_success, final_message)) {
      RCLCPP_ERROR(this->get_logger(),
                   "Failed to notify packing completion service for order_id=%d.",
                   request.order_id);
    }

    {
      std::lock_guard<std::mutex> guard(task_mutex_);
      task_in_progress_ = false;
    }
  }

  bool planAndExecuteSequence(const SequenceMsg& sequence, std::string& message) {
    constexpr double place_offset = 0.05;  // meters

    const auto target_pose = toPoseStamped(sequence, world_frame_);
    auto approach_pose = target_pose;
    approach_pose.pose.position.z += place_offset;
    auto retreat_pose = approach_pose;

    pauseBeforeStage(pause_before_place_move_, "before approach motion");

    std::string approach_message;
    if (!planAndExecuteArmPose(approach_pose, approach_message)) {
      message = std::string("Approach failed: ") + approach_message;
      return false;
    }
    pauseBeforeStage(pause_before_descent_, "before descent motion");

    std::string descend_message;
    if (!planAndExecuteArmPose(target_pose, descend_message)) {
      message = std::string("Descent failed: ") + descend_message;
      return false;
    }
    pauseBeforeStage(pause_before_release_, "before gripper release");

    std::string gripper_message;
    if (!planAndExecuteGripperNamed(gripper_open_pose_, gripper_message)) {
      message = std::string("Gripper release failed: ") + gripper_message;
      return false;
    }
    pauseBeforeStage(pause_before_retreat_, "before retreat motion");

    std::string retreat_message;
    if (!planAndExecuteArmPose(retreat_pose, retreat_message)) {
      message = std::string("Retreat failed: ") + retreat_message;
      return false;
    }
    pauseBeforeStage(pause_before_home_, "before returning home");

    std::string home_message;
    if (!planAndExecuteArmNamed(arm_home_pose_, home_message)) {
      message = std::string("Return home failed: ") + home_message;
      return false;
    }

    std::ostringstream oss;
    oss << "Approach: " << approach_message
        << " | Descent: " << descend_message
        << " | Gripper: " << gripper_message
        << " | Retreat: " << retreat_message
        << " | Return: " << home_message;
    message = oss.str();
    return true;
  }

  void pauseAfterSequence(std::int32_t sequence_id) {
    if (pause_between_sequences_.count() <= 0) {
      return;
    }
    RCLCPP_DEBUG(this->get_logger(),
                 "Sequence %d complete. Waiting %.1f seconds before continuing.",
                 sequence_id,
                 static_cast<double>(pause_between_sequences_.count()) / 1000.0);
    rclcpp::sleep_for(pause_between_sequences_);
  }

  void pauseBeforeStage(const std::chrono::milliseconds& duration, const char* context) {
    if (duration.count() <= 0) {
      return;
    }
    RCLCPP_DEBUG(this->get_logger(),
                 "Waiting %.1f seconds %s.",
                 static_cast<double>(duration.count()) / 1000.0,
                 context);
    rclcpp::sleep_for(duration);
  }

  bool planAndExecuteArmPose(const geometry_msgs::msg::PoseStamped& target_pose, std::string& message) {
    auto task = std::make_shared<mtc::Task>("arm_move_to_pose");

    try {
      task->loadRobotModel(shared_from_this(), "robot_description");
    } catch (const std::exception& e) {
      message = std::string("Failed to load robot model: ") + e.what();
      return false;
    }

    task->setProperty("group", arm_group_name_);
    task->setProperty("eef", gripper_group_name_);
    task->setProperty("ik_frame", gripper_frame_);

    auto pipeline_planner = std::make_shared<mtc::solvers::PipelinePlanner>(shared_from_this());
    task->add(std::make_unique<mtc::stages::CurrentState>("current_state"));

    auto move_to_pose = std::make_unique<mtc::stages::MoveTo>("move_to_sequence_pose", pipeline_planner);
    move_to_pose->setGroup(arm_group_name_);
    move_to_pose->setGoal(target_pose);
    task->add(std::move(move_to_pose));

    if (!arm_move_group_) {
      message = "Arm move group interface not initialised.";
      return false;
    }

    if (!planAndExecuteTask(task, *arm_move_group_, message)) {
      return false;
    }

    message = "Arm reached target pose.";
    return true;
  }

  bool planAndExecuteArmNamed(const std::string& pose_name, std::string& message) {
    auto task = std::make_shared<mtc::Task>("arm_move_named_pose");

    try {
      task->loadRobotModel(shared_from_this(), "robot_description");
    } catch (const std::exception& e) {
      message = std::string("Failed to load robot model: ") + e.what();
      return false;
    }

    task->setProperty("group", arm_group_name_);
    task->setProperty("eef", gripper_group_name_);
    task->setProperty("ik_frame", gripper_frame_);

    auto pipeline_planner = std::make_shared<mtc::solvers::PipelinePlanner>(shared_from_this());
    task->add(std::make_unique<mtc::stages::CurrentState>("current_state"));

    auto move_named = std::make_unique<mtc::stages::MoveTo>("move_to_named_pose", pipeline_planner);
    move_named->setGroup(arm_group_name_);
    move_named->setGoal(pose_name);
    task->add(std::move(move_named));

    if (!arm_move_group_) {
      message = "Arm move group interface not initialised.";
      return false;
    }

    if (!planAndExecuteTask(task, *arm_move_group_, message)) {
      return false;
    }

    std::ostringstream oss;
    oss << "Arm moved to '" << pose_name << "'.";
    message = oss.str();
    return true;
  }

  bool planAndExecuteGripperNamed(const std::string& pose_name, std::string& message) {
    auto task = std::make_shared<mtc::Task>("gripper_move_named_pose");

    try {
      task->loadRobotModel(shared_from_this(), "robot_description");
    } catch (const std::exception& e) {
      message = std::string("Failed to load robot model: ") + e.what();
      return false;
    }

    task->setProperty("group", arm_group_name_);
    task->setProperty("eef", gripper_group_name_);
    task->setProperty("ik_frame", gripper_frame_);

    auto joint_planner = std::make_shared<mtc::solvers::JointInterpolationPlanner>();
    task->add(std::make_unique<mtc::stages::CurrentState>("current_state"));

    auto move_gripper = std::make_unique<mtc::stages::MoveTo>("move_gripper_named_pose", joint_planner);
    move_gripper->setGroup(gripper_group_name_);
    move_gripper->setGoal(pose_name);
    task->add(std::move(move_gripper));

    if (!gripper_move_group_) {
      message = "Gripper move group interface not initialised.";
      return false;
    }

    if (!planAndExecuteTask(task, *gripper_move_group_, message, true)) {
      return false;
    }

    std::ostringstream oss;
    oss << "Gripper moved to '" << pose_name << "'.";
    message = oss.str();
    return true;
  }

  bool planAndExecuteTask(const std::shared_ptr<mtc::Task>& task,
                          moveit::planning_interface::MoveGroupInterface& move_group,
                          std::string& message,
                          bool allow_execute_failure = false) {
    active_tasks_.push_back(task);
    auto remove_task = rcpputils::make_scope_exit([this]() {
      if (!active_tasks_.empty()) {
        active_tasks_.pop_back();
      }
    });

    try {
      const auto plan_result = task->plan();
      if (plan_result != moveit::core::MoveItErrorCode::SUCCESS || task->numSolutions() == 0) {
        std::ostringstream oss;
        oss << "Planning failed with error code: " << plan_result.val;
        message = oss.str();
        logStageFailures(*task);
        return false;
      }
    } catch (const mtc::InitStageException& e) {
      message = std::string("Stage initialisation failed: ") + e.what();
      logTaskState(*task);
      return false;
    } catch (const std::exception& e) {
      message = std::string("Unexpected planning exception: ") + e.what();
      return false;
    }

    const auto& solution_ptr = task->solutions().front();
    if (publish_solutions_) {
      task->introspection().publishSolution(*solution_ptr);
    }

    std::vector<robot_trajectory::RobotTrajectoryConstPtr> segments;
    collectSubTrajectories(*solution_ptr, segments);

    if (segments.empty()) {
      message = "Solution contains no robot trajectories.";
      return false;
    }

    robot_trajectory::RobotTrajectory combined(segments.front()->getRobotModel(), move_group.getName());
    for (const auto& segment : segments) {
      if (!segment) {
        continue;
      }
      combined.append(*segment, 0.0);
    }

    moveit_msgs::msg::RobotTrajectory trajectory_msg;
    combined.getRobotTrajectoryMsg(trajectory_msg);

    auto execute_result = move_group.execute(trajectory_msg);
    const bool execution_failed = (execute_result != moveit::core::MoveItErrorCode::SUCCESS);
    if (execution_failed) {
      std::ostringstream oss;
      oss << "MoveGroup execution failed with code " << execute_result.val;
      message = oss.str();
      if (!allow_execute_failure) {
        return false;
      }
      RCLCPP_WARN(this->get_logger(),
                  "Execution failure for task '%s' ignored (gripper-only stage): %s",
                  task->name().c_str(), message.c_str());
    }

    move_group.stop();

    logStageSummary(*task);
    if (execution_failed && allow_execute_failure) {
      message = "Execution failed but was ignored per gripper policy.";
      return true;
    }

    message = "Task executed successfully.";
    return true;
  }

  void ensureMoveGroupInterfaces() {
    std::lock_guard<std::mutex> guard(move_group_mutex_);
    if (!arm_move_group_) {
      try {
        arm_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
          shared_from_this(), arm_group_name_);
        // Respect user-configurable arm speed limits.
        arm_move_group_->setMaxVelocityScalingFactor(arm_velocity_scaling_);
        arm_move_group_->setMaxAccelerationScalingFactor(arm_acceleration_scaling_);
      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create arm MoveGroupInterface: %s", e.what());
      }
    }

    if (!gripper_move_group_) {
      try {
        gripper_move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
          shared_from_this(), gripper_group_name_);
        // Respect user-configurable gripper speed limits.
        gripper_move_group_->setMaxVelocityScalingFactor(gripper_velocity_scaling_);
        gripper_move_group_->setMaxAccelerationScalingFactor(gripper_acceleration_scaling_);
      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create gripper MoveGroupInterface: %s", e.what());
      }
    }
  }

  void logStageFailures(const mtc::Task& task) const {
    RCLCPP_ERROR(this->get_logger(), "Stage failure summary:");
    for (std::size_t i = 0; i < task.stages()->numChildren(); ++i) {
      const auto* stage = task.stages()->operator[](i);
      RCLCPP_ERROR(this->get_logger(), "  %s -> %zu failures",
                   stage->name().c_str(), stage->failures().size());
    }
  }

  void logTaskState(const mtc::Task& task) const {
    std::ostringstream buffer;
    buffer << task;
    RCLCPP_ERROR(this->get_logger(), "Task state:\n%s", buffer.str().c_str());
  }

  void logStageSummary(const mtc::Task& task) const {
    RCLCPP_INFO(this->get_logger(), "Stage execution summary:");
    for (std::size_t i = 0; i < task.stages()->numChildren(); ++i) {
      const auto* stage = task.stages()->operator[](i);
      RCLCPP_INFO(this->get_logger(), "  %s -> %zu solutions, %zu failures",
                  stage->name().c_str(), stage->solutions().size(), stage->failures().size());
    }
  }

  bool delegateToVision(const PackeeMainStartMTC::Request& request,
                        const SequenceMsg& sequence,
                        std::string& message,
                        std::int32_t& total_detected) {
    if (!waitForService(vision_client_, vision_service_name_)) {
      message = "Vision service unavailable.";
      return false;
    }

    auto service_request = std::make_shared<ArmPickProduct::Request>();
    service_request->robot_id = request.robot_id;
    service_request->order_id = request.order_id;
    service_request->product_id = sequence.id;
    service_request->arm_side = "left";
    service_request->pose.x = 0.0F;
    service_request->pose.y = 0.0F;
    service_request->pose.z = 0.0F;
    service_request->pose.rx = 0.0F;
    service_request->pose.ry = 0.0F;
    service_request->pose.rz = 0.0F;

    beginMonitoring(sequence);
    auto monitoring_reset = rcpputils::make_scope_exit([this]() { endMonitoring(); });

    auto future = vision_client_->async_send_request(service_request);

    rclcpp::executors::SingleThreadedExecutor vision_executor;
    vision_executor.add_node(vision_proxy_node_);
    auto executor_guard = rcpputils::make_scope_exit([&vision_executor, this]() {
      vision_executor.remove_node(vision_proxy_node_);
    });
    const auto status = vision_executor.spin_until_future_complete(
      future, std::chrono::milliseconds(service_wait_timeout_ms_));

    if (status != rclcpp::FutureReturnCode::SUCCESS) {
      message = "Vision service timeout.";
      return false;
    }

    const auto response = future.get();
    total_detected = 0;
    message = response->message;
    return response->success;
  }

  bool sendPackingComplete(const PackeeMainStartMTC::Request& start_request,
                           bool success,
                           const std::string& message) {
    if (!waitForService(packing_complete_client_, packing_complete_service_name_)) {
      RCLCPP_ERROR(this->get_logger(),
                   "Packing completion service '%s' unavailable.",
                   packing_complete_service_name_.c_str());
      return false;
    }

    auto completion_request = std::make_shared<PackingComplete::Request>();
    completion_request->robot_id = start_request.robot_id;
    completion_request->order_id = start_request.order_id;
    completion_request->sequences = start_request.sequences;
    completion_request->success = success;
    completion_request->message = message;

    auto future = packing_complete_client_->async_send_request(completion_request);

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(vision_proxy_node_);
    auto guard = rcpputils::make_scope_exit([&executor, this]() {
      executor.remove_node(vision_proxy_node_);
    });

    const auto status = executor.spin_until_future_complete(
      future, std::chrono::milliseconds(service_wait_timeout_ms_));

    if (status != rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_ERROR(this->get_logger(),
                   "Packing completion service call timed out after %ld ms.",
                   service_wait_timeout_ms_);
      return false;
    }

    const auto response = future.get();
    if (!response->success) {
      RCLCPP_WARN(this->get_logger(),
                  "Packing completion service responded with success=false: %s",
                  response->message.c_str());
    } else {
      RCLCPP_INFO(this->get_logger(),
                  "Packing completion service acknowledged: %s",
                  response->message.c_str());
    }
    return response->success;
  }

  template<typename ClientT>
  bool waitForService(const ClientT& client, const std::string& service_name) {
    if (!client) {
      RCLCPP_ERROR(this->get_logger(), "Client for service '%s' is null.", service_name.c_str());
      return false;
    }

    if (!client->wait_for_service(std::chrono::milliseconds(service_wait_timeout_ms_))) {
      RCLCPP_ERROR(this->get_logger(), "Service '%s' not available after waiting.", service_name.c_str());
      return false;
    }
    return true;
  }

  void handleJointState(const sensor_msgs::msg::JointState::SharedPtr msg) {
    std::lock_guard<std::mutex> guard(state_mutex_);
    last_joint_state_ = msg;
    last_joint_state_time_ = this->get_clock()->now();
  }

  void monitorRobotState() {
    if (!monitoring_enabled_.load(std::memory_order_acquire)) {
      return;
    }

    sensor_msgs::msg::JointState::SharedPtr last_state;
    rclcpp::Time last_time;
    {
      std::lock_guard<std::mutex> guard(state_mutex_);
      if (!last_joint_state_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Monitoring robot state but no joint_states received yet.");
        return;
      }
      last_state = last_joint_state_;
      last_time = last_joint_state_time_;
    }

    const auto now = this->get_clock()->now();
    const double age_ms = (now - last_time).seconds() * 1000.0;

    if (age_ms > static_cast<double>(state_stale_timeout_ms_)) {
      RCLCPP_WARN(this->get_logger(),
                  "Robot joint state is stale while waiting for vision response (age=%.0f ms).",
                  age_ms);
    } else {
      RCLCPP_DEBUG(this->get_logger(),
                   "Monitoring robot state during vision hand-off: %s (age=%.0f ms).",
                   summariseJointState(*last_state).c_str(), age_ms);
    }
  }

  void beginMonitoring(const SequenceMsg& sequence) {
    {
      std::lock_guard<std::mutex> guard(monitor_mutex_);
      active_sequence_ = sequence.seq;
      active_product_id_ = sequence.id;
    }
    monitoring_enabled_.store(true, std::memory_order_release);
    RCLCPP_INFO(this->get_logger(),
                "Handed control to vision for sequence %d (product %d). Monitoring robot state only.",
                sequence.seq, sequence.id);
  }

  void endMonitoring() {
    monitoring_enabled_.store(false, std::memory_order_release);
    std::optional<std::int32_t> seq;
    std::optional<std::int32_t> product;
    {
      std::lock_guard<std::mutex> guard(monitor_mutex_);
      seq = active_sequence_;
      product = active_product_id_;
      active_sequence_.reset();
      active_product_id_.reset();
    }

    if (seq && product) {
      RCLCPP_INFO(this->get_logger(),
                  "Vision response received for sequence %d (product %d). Regaining control.",
                  *seq, *product);
    } else {
      RCLCPP_INFO(this->get_logger(),
                  "Vision response received. Regaining control.");
    }
  }

  std::string composeFailureMessage(const SequenceResult& result) const {
    std::ostringstream oss;
    oss << "Sequence " << result.sequence.seq << " (product " << result.sequence.id
        << ") failed: " << result.message
        << " (detected=" << result.total_detected << ")";
    return oss.str();
  }

  std::string composeSuccessMessage(const std::vector<SequenceResult>& results) const {
    std::ostringstream oss;
    bool first = true;
    for (const auto& result : results) {
      if (!first) {
        oss << "; ";
      }
      oss << "seq " << result.sequence.seq << " product " << result.sequence.id
          << ": " << result.message << " (detected=" << result.total_detected << ")";
      first = false;
    }
    return oss.str();
  }

  std::string summariseJointState(const sensor_msgs::msg::JointState& state) const {
    if (state.name.empty() || state.position.empty()) {
      return "{}";
    }

    const auto count = std::min(state.name.size(), state.position.size());
    const auto limit = std::min<std::size_t>(count, 3);

    std::ostringstream oss;
    oss << "{";
    for (std::size_t i = 0; i < limit; ++i) {
      if (i > 0) {
        oss << ", ";
      }
      oss << state.name[i] << "=" << std::fixed << std::setprecision(3) << state.position[i];
    }
    if (count > limit) {
      oss << ", ...";
    }
    oss << "}";
    return oss.str();
  }

  std::mutex task_mutex_;
  bool task_in_progress_{ false };//rviz display on/off

  std::string arm_group_name_;
  std::string gripper_group_name_;
  std::string gripper_frame_;
  std::string gripper_open_pose_;
  std::string arm_home_pose_;
  std::string world_frame_;
  std::string vision_service_name_;
  std::string packing_complete_service_name_;
  std::string joint_state_topic_;
  std::int64_t service_wait_timeout_ms_{ 3000 };
  std::int64_t monitor_interval_ms_{ 500 };
  std::int64_t state_stale_timeout_ms_{ 2000 };
  std::chrono::milliseconds pause_before_place_move_{ 3500 };
  std::chrono::milliseconds pause_before_descent_{ 3500 };
  std::chrono::milliseconds pause_before_release_{ 3500 };
  std::chrono::milliseconds pause_before_retreat_{ 3500 };
  std::chrono::milliseconds pause_before_home_{ 3500 };
  std::chrono::milliseconds pause_between_sequences_{ 3000 };
  bool publish_solutions_{ false };
  double arm_velocity_scaling_{ 0.3 };
  double arm_acceleration_scaling_{ 0.3 };
  double gripper_velocity_scaling_{ 0.3 };
  double gripper_acceleration_scaling_{ 0.3 };

  rclcpp::Service<PackeeMainStartMTC>::SharedPtr start_mtc_service_;
  rclcpp::Node::SharedPtr vision_proxy_node_;
  rclcpp::Client<ArmPickProduct>::SharedPtr vision_client_;
  rclcpp::Client<PackingComplete>::SharedPtr packing_complete_client_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
  rclcpp::TimerBase::SharedPtr monitor_timer_;
  std::future<void> worker_future_;
  std::vector<std::shared_ptr<mtc::Task>> active_tasks_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_move_group_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> gripper_move_group_;
  std::mutex move_group_mutex_;

  std::mutex state_mutex_;
  sensor_msgs::msg::JointState::SharedPtr last_joint_state_;
  rclcpp::Time last_joint_state_time_;

  std::atomic<bool> monitoring_enabled_{ false };
  std::mutex monitor_mutex_;
  std::optional<std::int32_t> active_sequence_;
  std::optional<std::int32_t> active_product_id_;

  void collectSubTrajectories(const mtc::SolutionBase& solution,
                              std::vector<robot_trajectory::RobotTrajectoryConstPtr>& output) const {
    if (const auto* sub = dynamic_cast<const mtc::SubTrajectory*>(&solution)) {
      if (sub->trajectory()) {
        output.push_back(sub->trajectory());
      }
      return;
    }

    if (const auto* sequence = dynamic_cast<const mtc::SolutionSequence*>(&solution)) {
      for (const auto* child : sequence->solutions()) {
        collectSubTrajectories(*child, output);
      }
      return;
    }

    if (const auto* wrapped = dynamic_cast<const mtc::WrappedSolution*>(&solution)) {
      if (wrapped->wrapped()) {
        collectSubTrajectories(*wrapped->wrapped(), output);
      }
    }
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int exit_code = 0;

  try {
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);
    auto node = std::make_shared<MtcOrchestrator>(options);

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("packee_mtc_orchestrator"),
                 "Unhandled exception: %s", e.what());
    exit_code = 1;
  }

  rclcpp::shutdown();
  return exit_code;
}
