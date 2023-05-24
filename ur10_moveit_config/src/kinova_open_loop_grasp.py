#! /usr/bin/env python3

import rospy, sys
import tf.transformations as tft
import numpy as np
# import kinova_msgs.msg
# import kinova_msgs.srv
import std_msgs.msg
import std_srvs.srv
import geometry_msgs.msg
from geometry_msgs.msg import Vector3, Quaternion, Transform, Pose, Point, Twist, Accel,PoseStamped
import moveit_commander
from moveit_msgs.msg import RobotState, Constraints, OrientationConstraint
import moveit_msgs.srv
import math
import robotiq_gripper


# from helpers.gripper_action_client import set_finger_positions
# from helpers.position_action_client import position_client, move_to_position

from helpers.transforms import current_robot_pose, publish_tf_quaterion_as_transform, convert_pose, publish_pose_as_transform
from helpers.covariance import generate_cartesian_covariance

MOVING = False  # Flag whether the robot is moving under velocity control.
ip = "192.168.0.131"

def robot_wrench_callback(msg):
    # Monitor wrench to cancel movement on collision.
    global MOVING
    if MOVING and msg.wrench.force.z < -2.0:
        MOVING = False
        rospy.logerr('Force Detected. Stopping.')

def gripper_grasp_width(gripper, width):
    control_instruction = int(255.0/85.0*width)
    gripper.move_and_wait_for_pos(255-control_instruction, 10, 10)
    print('grasp done')

def move_to_pose(pose):
    # Wrapper for move to position.
    # p = pose.position
    # o = pose.orientation
    # move_to_position([p.x, p.y, p.z], [o.x, o.y, o.z, o.w])

    target_pose = Pose()
    target_pose.position.x = pose.position.x
    target_pose.position.y = pose.position.y
    target_pose.position.z = pose.position.z
    target_pose.orientation.x = pose.orientation.x
    target_pose.orientation.y = pose.orientation.y
    target_pose.orientation.z = pose.orientation.z
    target_pose.orientation.w = pose.orientation.w
    arm.set_pose_target(target_pose)
    # Plan and execute the trajectory
    print('move start')
    plan = arm.go(wait=True)
    arm.stop()
    # arm.clear_pose_targets()
    print('move done')


    # target_pose = PoseStamped()
    # target_pose.header.frame_id = reference_frame
    # target_pose.header.stamp = rospy.Time.now()
    # target_pose.pose.position.x= pose.position.x
    # target_pose.pose.position.y= pose.position.y
    # target_pose.pose.position.z= pose.position.z
    # target_pose.pose.orientation.x = pose.orientation.x
    # target_pose.pose.orientation.y = pose.orientation.y
    # target_pose.pose.orientation.z = pose.orientation.z
    # target_pose.pose.orientation.w = pose.orientation.w
    # arm.set_joint_value_target(target_pose,True)
    # plan_=arm.plan()
    # if type(plan_) is tuple:
    #     # noetic
    #     success, plan, planning_time, error_code = plan_
    # arm.execute(plan)

def log_info(gripper):
    print(f"Pos: {str(gripper.get_current_position()): >3}  "
          f"Open: {gripper.is_open(): <2}  "
          f"Closed: {gripper.is_closed(): <2}  ")

def execute_grasp(gripper):
    # Execute a grasp.
    # global MOVING
    # global start_force_srv
    # global stop_force_srv

    # Get the positions. image positions from gg-cnn
    rospy.sleep(3) # let the other node find the position calculating
    msg = rospy.wait_for_message('/ggcnn/out/command', std_msgs.msg.Float32MultiArray)
    d = list(msg.data)
    print('command infor', d)
    pos=arm.get_current_pose().pose
    CURR_Z = pos.position.z
    # Calculate the gripper width.
    grip_width = d[4]
    # Convert width in pixels to mm.
    # 0.07 is distance from end effector (CURR_Z) to camera.
    # 0.1 is approx degrees per pixel for the realsense.
    g_width = 2 * ((CURR_Z + 0.07)) * np.tan(0.1 * grip_width / 2.0 / 180.0 * np.pi) * 1000
    # # Convert into motor positions.
    # g = min((1 - (min(g_width, 70)/70)) * (6800-4000) + 4000, 5500)
    # # set_finger_positions([g, g])
    print('finger width',g_width)
    # gripper.move_and_wait_for_pos(int(g_width), 255, 255)
    rospy.sleep(0.5)
    # Pose of the grasp (position only) in the camera frame.
    gp = geometry_msgs.msg.Pose()
    gp.position.x = d[0]
    gp.position.y = d[1]
    gp.position.z = d[2]
    gp.orientation.w = 1

    # # Convert to base frame, add the angle in (ensures planar grasp, camera isn't guaranteed to be perpendicular).
    gp_base = convert_pose(gp, 'camera_depth_optical_frame', 'base_link')

    q = tft.quaternion_from_euler(np.pi, 0, d[3])
    gp_base.orientation.x = q[0]
    gp_base.orientation.y = q[1]
    gp_base.orientation.z = q[2]
    gp_base.orientation.w = q[3]
    publish_pose_as_transform(gp_base, 'base_link', 'G', 0.5)
    # Offset for initial object pose.

    # move to obj over obj_over_height
    print('move to obj over obj_over_height')
    initial_offset = 0.40
    gp_base.position.z += initial_offset
    # print("gp_base", gp_base)
    # Disable force control, makes the robot more accurate.
    # stop_force_srv.call(kinova_msgs.srv.StopRequest())
    move_to_pose(gp_base)
    rospy.sleep(0.1)
    g_width = 70
    gripper_grasp_width(gripper, g_width)

    print('start to move down to grasp object')
    obj_over_height = -0.11
    gp_base.position.z += obj_over_height
    move_to_pose(gp_base)

    print('start grasping')
    g_width = 10
    gripper_grasp_width(gripper, g_width)
    print('grasping done')

    print('moving up')
    up_distance = 0.1
    gp_base.position.z += up_distance
    move_to_pose(gp_base)
    print('moving up done')

    #
    # # Start force control, helps prevent bad collisions.
    # start_force_srv.call(moveit_msgs.srv.StartRequest())
    #
    # rospy.sleep(0.25)
    #
    # # Reset the position
    # gp_base.position.z -= initial_offset
    #
    # # Flag to check for collisions.
    # MOVING = True
    #
    # # Generate a nonlinearity for the controller.
    # cart_cov = generate_cartesian_covariance(0)
    #
    # # Move straight down under velocity control.
    # velo_pub = rospy.Publisher('/base_link/in/cartesian_velocity', moveit_msgs.msg.PoseVelocity, queue_size=1)
    # while MOVING and CURR_Z - 0.02 > gp_base.position.z:
    #     dz = gp_base.position.z - CURR_Z - 0.03   # Offset by a few cm for the fingertips.
    #     MAX_VELO_Z = 0.08
    #     dz = max(min(dz, MAX_VELO_Z), -1.0*MAX_VELO_Z)
    #
    #     v = np.array([0, 0, dz])
    #     vc = list(np.dot(v, cart_cov)) + [0, 0, 0]
    #     # velo_pub.publish(kinova_msgs.msg.PoseVelocity(*vc))
    #     rospy.sleep(1/100.0)
    #
    # MOVING = False
    #
    # # close the fingers.
    # rospy.sleep(0.1)
    # set_finger_positions([8000, 8000])
    # rospy.sleep(0.5)
    #
    # # Move back up to initial position.
    # gp_base.position.z += initial_offset
    # gp_base.orientation.x = 1
    # gp_base.orientation.y = 0
    # gp_base.orientation.z = 0
    # gp_base.orientation.w = 0
    # move_to_pose(gp_base)
    #
    # stop_force_srv.call(moveit_msgs.srv.StopRequest())
    #
    # return

if __name__ == '__main__':
    nh=rospy.init_node('ggcnn_open_loop_grasp',anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.sleep(10)
    arm=moveit_commander.MoveGroupCommander('manipulator',wait_for_servers=30.0)
    reference_frame = 'base_link'
    arm.set_pose_reference_frame(reference_frame)
    arm.set_goal_joint_tolerance(0.001)
    arm.set_max_acceleration_scaling_factor(0.05)
    arm.set_max_velocity_scaling_factor(0.05)
    arm.set_planer_id = "RRTkConfigDefault"
    arm.set_planning_time(50)
    # Robot Monitors.
    wrench_sub = rospy.Subscriber('/wrench', geometry_msgs.msg.WrenchStamped, robot_wrench_callback, queue_size=1)
    # position_sub = rospy.Subscriber('/base_link/out/tool_pose', geometry_msgs.msg.PoseStamped, robot_position_callback, queue_size=1)
    # https://github.com/dougsm/rosbag_recording_services
    # start_record_srv = rospy.ServiceProxy('/data_recording/start_recording', std_srvs.srv.Trigger)
    # stop_record_srv = rospy.ServiceProxy('/data_recording/stop_recording', std_srvs.srv.Trigger)
    # Enable/disable force control.
    # start_force_srv = rospy.ServiceProxy('/base_link/in/start_force_control', moveit_msgs.srv.Start)
    # stop_force_srv = rospy.ServiceProxy('/base_link/in/stop_force_control', moveit_msgs.srv.Stop)

    # pose & angle test
    pos=arm.get_current_pose().pose
    print('current pose', pos)
    joints=arm.get_current_joint_values()
    print('current joints', np.array(joints)*180.0/math.pi)
    print('current joints', joints)

    #gripper test
    print("Creating gripper...")
    gripper = robotiq_gripper.RobotiqGripper()
    print("Connecting to gripper...")
    gripper.connect(ip, 63352)
    print("Activating gripper...")
    gripper.activate()
    print("Testing gripper...")
    gripper_grasp_width(gripper, 85)

    # Home position.
    home_joints = [1.2071189880371094, -1.5567658583270472, -1.5380070845233362, -1.6654790083514612, 1.5725183486938477, 0.004621791187673807]
    # Home pose
    home_pose = [-0.3914873222751472, -0.5714791260991916, 0.6681575664171412, 0.18301956387920323,0.9828197055899452, 0.021988522054863558, 0.009261233146959685]
    tcp1 = [-0.13851848713643922, -0.5736609613936515,0.44311404854058567, 0.09857153974906152, 0.9948931930812682, 0.01896440175118907, 0.010561125805382862]
    TCP = geometry_msgs.msg.Pose()
    TCP.position.x = tcp1[0]
    TCP.position.y = tcp1[1]
    TCP.position.z = tcp1[2]
    TCP.orientation.x = tcp1[3]
    TCP.orientation.y = tcp1[4]
    TCP.orientation.z = tcp1[5]
    TCP.orientation.w = tcp1[6]

    joint_release = [1.6262117624282837, -1.5525367895709437, -1.9304898420916956, -1.196756664906637, 1.572998046875, 0.0050779408775269985]
    # execute_grasp(gripper)

    arm.set_joint_value_target(home_joints)
    arm.go()
    rospy.sleep(0.5)

    while not rospy.is_shutdown():
        #move to home
        arm.set_joint_value_target(home_joints)
        arm.go()
        rospy.sleep(0.5)

        # set_finger_positions
        gripper_grasp_width(gripper, 85)
        rospy.sleep(3)

        # start_record_srv(std_srvs.srv.TriggerRequest())
        execute_grasp(gripper)
        rospy.sleep(0.5)

        # move to release
        arm.set_joint_value_target(joint_release)
        arm.go()
        move_to_pose(TCP)

        #open gripper
        gripper_grasp_width(gripper, 85)
        print('a loop done')


        # move_to_position([0, -0.38, 0.25], [0.99, 0, 0, np.sqrt(1-0.99**2)])
        # rospy.sleep(0.5)
        # stop_record_srv(std_srvs.srv.TriggerRequest())
        # raw_input('Press Enter to Complete')
