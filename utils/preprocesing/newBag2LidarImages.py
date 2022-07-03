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
bag_files_folder = "/home/potetsos/lagrinatorn/master/rellisBags"
image_topic = "/img_node"
output_dir = "/home/potetsos/lagrinatorn/master/rellisOutput/lidarImages"
bag_files = listdir(bag_files_folder)
#bag_file = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/stand_still_short.bag"

for bag_file in bag_files:
    print("\n")
    print("Bag file: ", bag_file)

    bag = rosbag.Bag(bag_files_folder +"/"+ bag_file, "r")
    bridge = CvBridge()
    
    topics = bag.get_type_and_topic_info()[1].keys()
    for topic in topics:
        print(topic)

    savePathBag = str(output_dir +"/"+ bag_file)[:-4]
    os.makedirs(savePathBag, exist_ok=True)

    for imageToptic in ['new_nearir_image', 'new_range_image', 'new_reflec_image', 'new_signal_image']:
    #for imageToptic in ['new_range_image']:
        savePathTopic = savePathBag + "/" + imageToptic
        os.makedirs(savePathTopic, exist_ok=True)
        print("Image topic: ", image_topic + "/" + imageToptic)
        count = 0
        for topic, msg, t in tqdm(bag.read_messages(topics=image_topic + "/" + imageToptic)): #tqdm
            cv_img = bridge.imgmsg_to_cv2(msg)
            #print(msg.header)
            #cv2.imwrite(savePathTopic + "/frame" + str(count).zfill(5) +".png", cv_img)
            #print(cv_img)
            #print(cv_img.shape)
            cv2.imwrite(f"{savePathTopic}/{msg.header.stamp.secs}_{str(msg.header.stamp.nsecs)[0:3]}.png", cv_img)
            count += 1

            

            #if(count == 1):
            #    firstTime = t
            #lastTime = t

            

        #totTime = lastTime - firstTime
        #print("Total time:", totTime)
        #print("Total time sec:", totTime/(10**9))
        #print("Number of measurements:", count)
        #print("Avg time:", totTime/count)
        #print("Avg time sec:", (int(str(totTime))/count)/(10**9))
        #print("First", firstTime)
        #print("Last", lastTime)

        #break
            #if (count > 10):
            #     break
        
    bag.close()
    #break

