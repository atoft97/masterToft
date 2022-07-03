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
import numpy as np

#parser = argparse.ArgumentParser(description='Extract lidar images from a ROS-bags in a folder.')
#parser.add_argument("bag_files_folder", help="Input ROS bag (../data/bagFiles)")
#parser.add_argument("output_dir", help="Output directory (../data/lidarImages)")
#parser.add_argument("image_topic", help="Image topic (/ugv_sensors/lidar/image)")

#args = parser.parse_args()

#print("Extract images from %s on topic %s into %s" % (args.bag_file, args.image_topic, args.output_dir))
input_toptic = "/ugv_sensors/lidar/image"
bag_files_folder = "/home/potetsos/lagrinatorn/master/bagfiler/newBags"
bag_files = listdir(bag_files_folder)
#bag_files = ["plains_drive.bag"]
#bag_file = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/stand_still_short.bag"

#bag2Name = {"plains_drive.bag": "plainsDrive", "plains_drive2.bag": "plainsDrive", "road_drive.bag": "roadDrive", "road_drive2.bag": "roadDrive2", "rock_quarry_drive.bag": "rockQuarryDrive", "rock_quarry_into_woods_drive.bag": "rockQuarryIntoWoodsDrive", "stand_still_short.bag": "stille"}
output_dir = "/home/potetsos/skule/2022/masterCode/masterToft/data/lidarImages16"

#bag_files = listdir(args.bag_files_folder)
#bag_file = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/stand_still_short.bag"

for bag_file in bag_files:
    print("\n")
    print("Bag file: ", bag_file)

    try:
        bag = rosbag.Bag(bag_files_folder +"/"+ bag_file, "r")
    except:
        print("feil med baggen")
        continue

    bridge = CvBridge()
    
    topics = bag.get_type_and_topic_info()[1].keys()
    #for topic in topics:
    #    print(topic)

    savePathBag = str(output_dir +"/"+ bag_file)[:-4]
    os.makedirs(savePathBag, exist_ok=True)

    for imageToptic in ['nearir_image', 'range_image', 'reflec_image', 'signal_image']:
        savePathTopic = savePathBag + "/" + imageToptic
        os.makedirs(savePathTopic, exist_ok=True)
        print("Image topic: ", imageToptic)
        count = 0
        for topic, msg, t in tqdm(bag.read_messages(topics=input_toptic + "/" + imageToptic)): #tqdm
            cv_img = bridge.imgmsg_to_cv2(msg)
            #cv_img.astype(np.uint16)
            #print(cv_img)
            count += 1
            cv2.imwrite(savePathTopic + "/" + bag_file[:-4] + "_" + str(count).zfill(5) +".png", cv_img)
            if(count == 1):
                firstTime = t
            lastTime = t

        '''
        totTime = lastTime - firstTime
        print("Total time:", totTime)
        print("Total time sec:", totTime/(10**9))
        print("Number of measurements:", count)
        print("Avg time:", totTime/count)
        print("Avg time sec:", (int(str(totTime))/count)/(10**9))
        print("First", firstTime)
        print("Last", lastTime)
        '''
    
    bag.close()

    
    

