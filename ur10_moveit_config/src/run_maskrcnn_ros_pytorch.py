#! /usr/bin/env python3

import time
import os
from os import path
import numpy as np
import torch
import cv2
import scipy.ndimage as ndimage
from skimage.draw import disk
from skimage.feature import peak_local_max



import rospy, sys
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
import moveit_commander

import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
matplotlib.use('TkAgg')
from detectron2.structures import Instances
import random

bridge = CvBridge()

# Check if CUDA is available and choose device accordingly
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


# Load the Network.
MODEL_FILE = 'models/epoch_42_iou_0.73'
model = torch.load(MODEL_FILE, map_location=device)
model.eval()

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import os

rospy.init_node('ggcnn_detection')

cfg = get_cfg()
CLASS_NAMES =["A","B","C"]

# Output publishers.
grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
grasp_plain_pub = rospy.Publisher('ggcnn/img/grasp_plain', Image, queue_size=1)
depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)

# Initialise some globals.
prev_mp = np.array([150, 150])
ROBOT_Z = 0

# Get the camera parameters
camera_info_msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
print(camera_info_msg)
K = camera_info_msg.K
fx = K[0]
cx = K[2]
fy = K[4]
cy = K[5]


filter_num = 2
ment_array = [[0]*6 for _ in range(filter_num)]
callback_counter = 0

# xy offset

x_offset = -0.02
y_offset = -0.01

# Execution Timing
class TimeIt:
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.print_output = False

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        self.t1 = time.time()
        if self.print_output:
            print('%s: %s' % (self.s, self.t1 - self.t0))


# def robot_pos_callback(data):
#     global ROBOT_Z
#     ROBOT_Z = data.pose.position.z

def return_tip():
    pos=arm.get_current_pose().pose
    # tip=[pos.position.x,pos.position.y,pos.position.z]
    # global ROBOT_Z
    Z = pos.position.z
    return Z

def depth_callback(depth_message):
    global ment_array
    global callback_counter
    global model
    global graph
    global prev_mp
    # global ROBOT_Z
    global fx, cx, fy, cy
    ROBOT_Z = return_tip()
    print('broadcasting.. tip_z', ROBOT_Z)

    with TimeIt('Crop'):
        print('croping..')
        aligned_depth = bridge.imgmsg_to_cv2(depth_message)
        depth_data = np.asanyarray(aligned_depth, dtype="float16")

        array = np.copy(depth_data)
        # Create x and y coordinates
        x = np.arange(0, array.shape[1])
        y = np.arange(0, array.shape[0])
        xx, yy = np.meshgrid(x, y)
        # Get valid (non-zero) and invalid (zero) indices
        valid_idx = np.where(array > 1)
        invalid_idx = np.where(array <= 1)
        # Interpolate only the invalid values
        interp_values = interpolate.griddata(valid_idx, array[valid_idx],
                            invalid_idx, method='nearest')
        # Create a copy of the original image and replace the invalid values
        # with interpolated values
        inpainted = np.copy(array)
        inpainted[invalid_idx] = interp_values
        inpainted = inpainted.astype(np.float32)

        depth_copy_uncropped = np.copy(inpainted)

    with TimeIt('Inpaint'):
        print('Inpaint..')

    with TimeIt('Calculate Depth'):
        print('Calculate Depth..')

        depth_center_list = []
        box_center_x = []
        box_center_y = []
        for i, box in enumerate(boxes):
           x1, y1, x2, y2 = box.astype(int)  # Convert the coordinates to integers
           # Figure out roughly the depth in mm of the part between the grippers for collision avoidance.

           depth_center_tmp = depth_copy_uncropped[y1:y2, x1:x2].flatten()
           depth_center_tmp.sort()
           depth_center_tmp = depth_center_tmp[:10].mean() # * 1000.0
           depth_center_list.append(depth_center_tmp)
           box_center_x.append(int((x1+x2)/2))
           box_center_y.append(int((y1+y2)/2))

        if len(depth_center_list) > 0:
            depth_center = sum(depth_center_list)/len(depth_center_list)
        else:
            print('no obj detected')


    with TimeIt('Inference'):
        print('Inference..')


    with TimeIt('Trig'):
        print('Trig..')
        # Calculate the angle map.
        ang_out = 0
        width_out =0

        # print('ang', 'width', ang_out, width_out)

    with TimeIt('Filter'):
        print('Filter..')
        # Filter the outputs.
    with TimeIt('Control'):
        print('Control..')
        # Calculate the best pose from the camera intrinsics.
        maxes = None

        ALWAYS_MAX = False  # Use ALWAYS_MAX = True for the open-loop solution.

        # if ROBOT_Z > 0.34 or ALWAYS_MAX:  # > 0.34 initialises the max tracking when the robot is reset.
            # Track the global max.

        point_depth = depth_center/1000.0

        max_pixel = [0,0]
        max_pixel[1] = box_center_x[0]
        max_pixel[0] = box_center_y[0]
        # These magic numbers are my camera intrinsic parameters.
        x = (max_pixel[1] - cx)/(fx) * point_depth
        y = (max_pixel[0] - cy)/(fy) * point_depth
        z = point_depth

        if np.isnan(z):
            return

    with TimeIt('Draw'):
        print('Draw..')
        # Draw grasp markers on the points_out and publish it. (for visualisation)
        grasp_img = np.zeros((300, 300, 3), dtype=np.uint8)
        # grasp_img[:,:,2] = (points_out * 255.0)

        grasp_img_plain = grasp_img.copy()

        # centre = (prev_mp[0], prev_mp[1])
        centre = (min(max(prev_mp[0], 5), 294), min(max(prev_mp[1], 5), 294))
        rr, cc = disk(centre, 5)
        grasp_img[rr, cc, 0] = 0
        grasp_img[rr, cc, 1] = 255
        grasp_img[rr, cc, 2] = 0

    with TimeIt('Publish'):
        print('Publish..')
        # Publish the output images (not used for control, only visualisation)
        # grasp_img = bridge.cv2_to_imgmsg(grasp_img)
        # grasp_img.header = depth_message.header
        # grasp_pub.publish(grasp_img)
        # grasp_img_plain = bridge.cv2_to_imgmsg(grasp_img_plain)
        # grasp_img_plain.header = depth_message.header
        # grasp_plain_pub.publish(grasp_img_plain)
        # depth_pub.publish(bridge.cv2_to_imgmsg(depth_crop))
        # ang_pub.publish(bridge.cv2_to_imgmsg(ang_out))
        # Output the best grasp pose relative to camera.
        cmd_msg = Float32MultiArray()
        # cmd_msg.data = [x, y, z, ang, width, depth_center]

        if z>0.28 and z<4.2:
            cmd_msg.data = [x+x_offset, y+y_offset, z, np.pi, 0, depth_center]
            print('cmd_msg.data', cmd_msg.data)
            cmd_pub.publish(cmd_msg)


# Define the callback for the camera subscriber
def image_callback(img_msg):
    # Convert ROS Image message to OpenCV image
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

    # Perform object detection on the image
    outputs = predictor(cv_image)

    # TODO: Add code to handle the outputs of the object detection model, e.g.,
    # visualize the results on the image, publish the results to a ROS topic, etc.

    MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes = CLASS_NAMES
    class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_classes

    # Desired class
    desired_class_name = 'B'
    desired_class_index = class_names.index(desired_class_name)
    # Create a mask for instances of the desired class
    class_mask = outputs["instances"].pred_classes == desired_class_index

        # Filter the instances using the mask
    desired_instances = Instances(outputs["instances"].image_size, **{k: v[class_mask] for k, v in outputs["instances"].get_fields().items()})

    global boxes
    # Get bounding boxes and calculate centers
    if desired_instances.has("pred_boxes"):
        boxes = desired_instances.pred_boxes.tensor.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            print(f"Object: {desired_class_name}, Bounding box: ({x1}, {y1}), ({x2}, {y2}), Center: ({center_x}, {center_y})")

    # Visualize the results on the image
    v = Visualizer(cv_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1)
    out = v.draw_instance_predictions(desired_instances.to("cpu"))
    visualized_image = out.get_image()[:, :, ::-1]
    cv2.imshow('Visualized image', visualized_image)
    cv2.waitKey(1)


# rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
# depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_callback, queue_size=1)
# robot_pos_sub = rospy.Subscriber('/base_link/out/tool_pose', PoseStamped, robot_pos_callback, queue_size=1)


if __name__ == "__main__":

    # nh=rospy.init_node('agent',anonymous=True)
    moveit_commander.roscpp_initialize(sys.argv)
    arm=moveit_commander.MoveGroupCommander('manipulator')
    reference_frame = 'base_link'
    arm.set_pose_reference_frame(reference_frame)
    arm.set_goal_joint_tolerance(0.001)
    arm.set_max_acceleration_scaling_factor(0.02)
    arm.set_max_velocity_scaling_factor(0.02)
    arm.set_planer_id = "RRTkConfigDefault"
    arm.set_planning_time(50)

        # Initialize the object detection model
    config_file =  "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TEST = ("my_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.WEIGHTS = os.path.join('../output/', "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    global predictor
    predictor = DefaultPredictor(cfg)


    # Subscribe to the RealSense camera image topic
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_callback, queue_size=1)


    while not rospy.is_shutdown():
        rospy.spin()
        print('test')

