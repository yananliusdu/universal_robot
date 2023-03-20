#!/usr/bin/env python3

from moveit_msgs.msg import RobotState, Constraints, OrientationConstraint
import rospy, sys
import geometry_msgs.msg
from geometry_msgs.msg import Vector3, Quaternion, Transform, Pose, Point, Twist, Accel,PoseStamped
import rospy
from std_msgs.msg import String,Float64MultiArray,MultiArrayDimension
import numpy as np
import math
import moveit_commander

import time

import cv2


class robot():
    def __init__(self,arm):
        self.arm=arm
    def reset(self):
        pos1=[1.2675,-0.1259-math.pi/2,-1.754,-1.2617,1.5608,0]
        self.arm.set_joint_value_target(pos1)
        self.arm.go()
        self.sem_to_pos={'one':0,'two':1,'three':2,'four':3,'five':4,'six':6}
        self.label_to_pos={'0':None,'1':[1,0],'2':[2,0],'3':[3,0],'4':[4,0],'5':[5,0],'6':[6,0],
                           '7':[1,0],'8':[2,0],'9':[3,0],'10':[4,0],'11':[5,0],'12':[6,0],
                           '13':[1,0],'14':[2,0],'15':[3,0],'16':[4,0],'17':[5,0],'18':[6,0]}
        self.label_to_object={'0':None,'1':[0,1],'2':[0,1],'3':[0,1],'4':[0,1],'5':[0,1],'6':[0,1],
                           '7':[0,2],'8':[0,2],'9':[0,2],'10':[0,2],'11':[0,2],'12':[0,2],
                           '13':[0,3],'14':[0,3],'15':[0,3],'16':[0,3],'17':[0,3],'18':[0,3]}
        self.graph=[]
        self.link=[]
    def pick_pos(self,index,num):
        pos1={'0':np.array([85.98,-83.00,-124.72,-63.52,89.12,0])*math.pi/180,
              '1':np.array(),
              '2':np.array(),
              }
        return pos1[str(index)][num]
    def place_pos(self,index):
        pos1={'0':np.array([85.98,-83.00,-124.72,-63.52,89.12,0])*math.pi/180,
            '1':np.array([85.98,-83.00,-124.72,-63.52,89.12,0])*math.pi/180,
              '2':np.array(),
              '3':np.array(),
              '4':np.array(),
              '5':np.array()}
        return pos1[str(index)]
    def graph(self,label):
        self.label_to_link[label]
        self.label_to_object[label]
    def robot_control(self):
        pos=self.pick_pos(1, 2)
        pos1=pos
        pos1=pos1[4]+32
        arm.set_joint_value_target(pos1)
        arm.go()
        gripper.move_and_wait_for_pos(70, 255, 255)
        arm.set_joint_value_target(pos)
        arm.go()
        gripper.move_and_wait_for_pos(150, 255, 255)
        arm.set_joint_value_target(pos1)
        arm.go()
        
        pos=self.place_pos(1)
        pos1=pos
        pos1=pos1[4]+32
        arm.set_joint_value_target(pos1)
        arm.go()
        arm.set_joint_value_target(pos)
        arm.go()
        gripper.move_and_wait_for_pos(70, 255, 255)
def return_tip():
    pos=arm.get_current_pose().pose
   
    tip=[pos.position.x,pos.position.y,pos.position.z]
    joints=arm.get_current_joint_values()
    joint1=[ -42.25039272,  -81.30925707, -102.48127163,  -82.62745284,   85.61020577, -167.64634112]
    joint1=np.array(joint1)*math.pi/180
    #print(np.array(joints)*180/math.pi)
    #joints[5]=joints[5]+5*math.pi/180
    arm.set_joint_value_target(joint1)
    arm.go()
    print(np.array(joints)*180/math.pi)
def move(angle):
    target_pose = PoseStamped()
    target_pose.header.frame_id = reference_frame
    target_pose.header.stamp = rospy.Time.now()
    target_pose.pose.position.x=float(angle[0])
    target_pose.pose.position.y=float(angle[1])
    target_pose.pose.position.z=float(angle[2])
    target_pose.pose.orientation.x = 0
    target_pose.pose.orientation.y = 1
    target_pose.pose.orientation.z = 0
    target_pose.pose.orientation.w = 0
    
    arm.set_joint_value_target(target_pose,True)
    plan_=arm.plan()
    if type(plan_) is tuple:
        # noetic
        success, plan, planning_time, error_code = plan_
    arm.execute(plan)
def reset():
    joints=arm.get_current_joint_values()
    print(joints)
   
    pos1=np.array([85.27,-41.81,-124.34,-99.20,89.99,0])*math.pi/180
    print(pos1)
    arm.set_joint_value_target(pos1)
    arm.go()
def move_robot(pos1,grab_pos):
    gripper.move_and_wait_for_pos(70, 255, 255)
    move(pos1)
    
   
    
    
    pos1[2]=pos1[2]-0.04
    move(pos1)
    gripper.move_and_wait_for_pos(150, 255, 255)
    pos1[2]=pos1[2]+0.18
    move(pos1)
    
    
    
    
    grab_pos[2]=grab_pos[2]+0.18
    move(grab_pos)
    # grab_pos[2]=0.365
    # grab_pos[2]=0.415
    grab_pos[2]=0.385
    move(grab_pos)
    
    gripper.move_and_wait_for_pos(70, 255, 255)
    grab_pos[2]=0.51
    move(grab_pos)
    reset()

ip = "192.168.0.131"
if __name__ == "__main__":
    
    
    nh=rospy.init_node('agent',anonymous=True)
   
    moveit_commander.roscpp_initialize(sys.argv)
    arm=moveit_commander.MoveGroupCommander('manipulator')
    reference_frame = 'base_link'
    arm.set_pose_reference_frame(reference_frame)
    arm.set_goal_joint_tolerance(0.001)
    arm.set_max_acceleration_scaling_factor(0.02)
    arm.set_max_velocity_scaling_factor(0.02)
    arm.set_planer_id = "RRTkConfigDefault"
    arm.set_planning_time(50)
    
    return_tip()
    
