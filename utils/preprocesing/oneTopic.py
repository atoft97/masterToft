#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse
import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from os import listdir
from tqdm import tqdm

#parser = argparse.ArgumentParser(description='Extract lidar images from a ROS-bags in a folder.')
#parser.add_argument("bag_files_folder", help="Input ROS bag (../data/bagFiles)")
#parser.add_argument("output_dir", help="Output directory (../data/lidarImages)")
#parser.add_argument("image_topic", help="Image topic (/ugv_sensors/lidar/image)")

#args = parser.parse_args()

#print("Extract images from %s on topic %s into %s" % (args.bag_file, args.image_topic, args.output_dir))

#bag_file = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/stand_still_short.bag"
bag_file = "/home/potetsos/lagrinatorn/master/bagfiler/andreTur/2022-05-06-15-45-00_0.bag"


print("\n")
print("Bag file: ", bag_file)

bag = rosbag.Bag(bag_file, "r")
bridge = CvBridge()

topics = bag.get_type_and_topic_info()[1].keys()
#for topic in topics:
#    print(topic)


print("\n")
for topicI in ['/tf_static']:
    for topic, msg, t in tqdm(bag.read_messages(topics=topicI)): #tqdm

        #print(msg)
        #print(msg)
        #print(msg.transforms[0].header.frame_id)
        #utgangspunkt = msg.transforms[0].header.frame_id
        #child = msg.transforms[0].child_frame_id
        
        #if (child !="P" and child !="B" and child !="N" and child !="Q"):
        #    print(child)
        print(msg)
        #if (utgangspunkt !="N" and utgangspunkt != "E" and utgangspunkt != "Lf"):
        #    print(msg)
            #break
        #break


bag.close()


