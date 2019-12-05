#!/usr/bin/env python
from __future__ import print_function

import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
import tensorflow as tf
import time
roslib.load_manifest('rbx1_apps')
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from math import copysign
import point_cloud2

class Follower():
    def __init__(self):
        #rospy.init_node("follower")

        # Set the shutdown function (stop the robot)
        rospy.on_shutdown(self.shutdown)

        # The dimensions (in meters) of the box in which we will search
        # for the person (blob). These are given in camera coordinates
        # where x is left/right,y is up/down and z is depth (forward/backward)

        # This is cropping the point cloud screen so we can read them from the parameter server instead of hard-coding values each time
        self.min_x = rospy.get_param("~min_x", -0.2)
        self.max_x = rospy.get_param("~max_x", 0.2)
        self.min_y = rospy.get_param("~min_y", -0.3)
        self.max_y = rospy.get_param("~max_y", 0.5)
        self.max_z = rospy.get_param("~max_z", 1.2)

        # The goal distance (in meters) to keep between the robot and the person
        self.goal_z = rospy.get_param("~goal_z", 0.6)

        # How far away from the goal distance (in meters) before the robot reacts
        self.z_threshold = rospy.get_param("~z_threshold", 0.05)

        # How far away from being centered (x displacement) on the person
        # before the robot reacts
        self.x_threshold = rospy.get_param("~x_threshold", 0.05)

        # How much do we weight the goal distance (z) when making a movement
        self.z_scale = rospy.get_param("~z_scale", 1.0)

        # How much do we weight x-displacement of the person when making a movement
        self.x_scale = rospy.get_param("~x_scale", 2.5)

        # The maximum rotation speed in radians per second
        self.max_angular_speed = rospy.get_param("~max_angular_speed", 2.0)

        # The minimum rotation speed in radians per second
        self.min_angular_speed = rospy.get_param("~min_angular_speed", 0.0)

        # The max linear speed in meters per second
        self.max_linear_speed = rospy.get_param("~max_linear_speed", 0.3)

        # The minimum linear speed in meters per second
        self.min_linear_speed = rospy.get_param("~min_linear_speed", 0.1)

        # Publisher to control the robot's movement
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel_mux/input/navi', Twist)

        #rospy.Subscriber('camera/rgb/image_raw', PointCloud2, self.set_cmd_vel)

        # Wait for the pointcloud topic to become available
        #rospy.wait_for_message('point_cloud', PointCloud2)

    def set_cmd_vel(self, xh,yh,zh,left):
        # Initialize the centroid coordinates point count
        pt_x = xh
	pt_y = yh
	pt_z = zh
	n = 1

        # Read in the x, y, z coordinates of all points in the cloud
        """for point in point_cloud2.read_points(msg, skip_nans=True):
            pt_x = point[0]
            pt_y = point[1]
            pt_z = point[2]"""

	 # Keep only those points within our designated boundaries and sum them up
	"""
	 if -pt_y > self.min_y and -pt_y < self.max_y and  pt_x < self.max_x and pt_x > self.min_x and pt_z < self.max_z:
	     x += pt_x
     	     y += pt_y
	     z += pt_z
	"""
		

        # Stop the robot by default
        move_cmd = Twist()

        # If we have points, compute the centroid coordinates
        if n:
            x =  pt_x
            y = pt_y
            z = pt_z

            # Check our movement thresholds
            if (abs(z - self.goal_z) > self.z_threshold) or (abs(x) > self.x_threshold):
                # Compute the linear and angular components of the movement
                linear_speed = (z - self.goal_z) * self.z_scale
		if(x>=left and x<= left*2):
			x=0
		if(x>=2*left):
			x*=-1
		

                angular_speed = x * self.x_scale

                # Make sure we meet our min/max specifications
                linear_speed = copysign(max(self.min_linear_speed,
                                            min(self.max_linear_speed, abs(linear_speed))), linear_speed)
                angular_speed = copysign(max(self.min_angular_speed,
                                             min(self.max_angular_speed, abs(angular_speed))), angular_speed)

                move_cmd.linear.x = linear_speed/4
                move_cmd.angular.z = angular_speed/6

	#print( linear_speed , angular_speed)
        # Publish the movement command
        self.cmd_vel_pub.publish(move_cmd)


    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()


        #print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image)
    self.move_pub = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=1)
    self.move_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depthcb)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.callback,queue_size=30)

  radius = -1
  center = [-1,-1]
  left = -1
  model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
  odapi = DetectorAPI(path_to_ckpt=model_path)
  f = Follower()
  width = 0
  threshold = 0.7
  def depthcb(self,data):
		try:
		 cv_image = self.bridge.imgmsg_to_cv2(data, "32FC1")

		except CvBridgeError as e:
		   	 print(e)
		#	print(self.center[0],self.left)
	
		(rows,cols) = cv_image.shape
		
		
		if(self.center[0]!= -1 and self.radius !=-1):
			if(self.radius>20 and self.left!=-1):
				#print(self.width,cols)
				#print(self.center[0],self.center[1],self.left)
				dist = cv_image[self.center[1]][self.center[0]]
				
				print(dist)
				if(dist>=1200):
					self.f.set_cmd_vel(self.center[0],self.center[1],dist,self.left)
	
				
		"""
				if(self.center[0]>self.left and self.center[0]<2*self.left):
					dist = cv_image[self.center[0]][self.center[1]]
					print(dist)
					if(self.center[0]<=self.left):
							msg = Twist()
							msg.angular.z = 0.3
							self.move_pub.publish(msg)
					elif(self.center[0]>=(self.left*2)):
							msg = Twist()
							msg.angular.z = -0.3
							self.move_pub.publish(msg)
					else():
							msg = Twist()
							msg.linear.x = 0.2
							self.move_pub.publish(msg)
		"""
	
		
				
					
  def callback(self,data):
	
		try:
		 cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		 
		except CvBridgeError as e:
		   	 print(e)
		
		(rows,cols,channels) = cv_image.shape
		self.left = cols//3
		self.width = cols;
		boxes, scores, classes, num = self.odapi.processFrame(cv_image)
		
		check = 0
		for i in range(len(boxes)):
		    # Class 1 represents human
		    if classes[i] == 1 and scores[i] > self.threshold:
			box = boxes[i]
			cv2.rectangle(cv_image,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
			self.center[0] = box[1]+abs(box[1]-box[3])//2
			self.center[1] = box[0]+abs(box[0]-box[2])//2
	   		self.radius=21
			check =1

		
		if(check==0):
			#print(self.radius)
			self.radius=-1
			#self.f.shutdown()
	
		
		
		cv2.imshow("Frame", cv_image)
		key = cv2.waitKey(1)

		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		except CvBridgeError as e:
			print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
