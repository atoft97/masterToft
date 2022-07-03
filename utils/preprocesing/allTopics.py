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

parser = argparse.ArgumentParser(description='Extract lidar images from a ROS-bags in a folder.')
parser.add_argument("bag_files_folder", help="Input ROS bag (../data/bagFiles)")
parser.add_argument("output_dir", help="Output directory (../data/lidarImages)")
parser.add_argument("image_topic", help="Image topic (/ugv_sensors/lidar/image)")

args = parser.parse_args()

#print("Extract images from %s on topic %s into %s" % (args.bag_file, args.image_topic, args.output_dir))

bag_files = listdir(args.bag_files_folder)
#bag_file = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/stand_still_short.bag"

for bag_file in [bag_files[-1]]:
    print("\n")
    print("Bag file: ", bag_file)

    bag = rosbag.Bag(args.bag_files_folder +"/"+ bag_file, "r")
    bridge = CvBridge()
    
    topics = bag.get_type_and_topic_info()[1].keys()
    for topic in topics:
        print(topic)

    print("\n")
    for topicI in topics:
        for topic, msg, t in tqdm(bag.read_messages(topics=topicI)): #tqdm
            if (topic not in ["/ugv_sensors/camera/color/image", "/ugv_sensors/camera/color/image/compressed", "/ugv_sensors/camera/color/image/theora", "/ugv_sensors/camera/left/image", 
                "/ugv_sensors/camera/left/right", "/ugv_sensors/camera/left/image/compressed", "/ugv_sensors/camera/right/image/compressed", "/ugv_sensors/camera/right/image/theora",
                "/ugv_sensors/camera/left/image/theora", "/ugv_sensors/camera/right/image", "/ugv_sensors/lidar/cloud/points", "/ugv_sensors/lidar/driver/lidar_packets",
                "/ugv_sensors/lidar/image/nearir_image", '/ugv_sensors/lidar/image/range_image', '/ugv_sensors/lidar/image/reflec_image', '/ugv_sensors/lidar/image/signal_image']):
                print("Image topic: ", topic)
                #print(msg)

                try:
                    print(msg.data)
                except:
                    print(msg)

                print("\n")




            break


    bag.close()
    break

