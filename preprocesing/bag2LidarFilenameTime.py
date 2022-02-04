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
from csv import DictWriter

parser = argparse.ArgumentParser(description='Extract lidar images from a ROS-bags in a folder.')
parser.add_argument("bag_files_folder", help="Input ROS bag (../data/bagFiles)")
parser.add_argument("output_dir", help="Output directory (../data/lidarImages)")
parser.add_argument("image_topic", help="Image topic (/ugv_sensors/lidar/image)")

args = parser.parse_args()

#print("Extract images from %s on topic %s into %s" % (args.bag_file, args.image_topic, args.output_dir))

bag_files = listdir(args.bag_files_folder)
#bag_file = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/stand_still_short.bag"

def write_to_csv(new_data, fileName):
    with open(fileName, 'a') as f_object:
        writer_object = DictWriter(f_object, fieldnames=['time', 'filename'])
        writer_object.writerow(new_data)
        f_object.close()


for bag_file in bag_files:
    print("\n")
    print("Bag file: ", bag_file)

    bag = rosbag.Bag(args.bag_files_folder +"/"+ bag_file, "r")
    bridge = CvBridge()
    
    topics = bag.get_type_and_topic_info()[1].keys()
    for topic in topics:
        print(topic)

    savePathBag = str(args.output_dir)
    os.makedirs(savePathBag, exist_ok=True)

    csvPath = savePathBag + "/"+ bag_file[:-4] + ".csv"

    if (os.path.exists(csvPath)):
        os.remove(csvPath)
    write_to_csv({'time': 'time', 'filename': 'filename'}, csvPath)

    for imageToptic in ['nearir_image']:
        savePathTopic = savePathBag + "/" + imageToptic
        #os.makedirs(savePathTopic, exist_ok=True)
        print("Image topic: ", imageToptic)
        count = 0
        for topic, msg, t in tqdm(bag.read_messages(topics=args.image_topic + "/" + imageToptic)): #tqdm
            #cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            #cv2.imwrite(savePathTopic + "/frame" + str(count).zfill(5) +".png", cv_img)
            
            timePOS = {'time': t,'filename': bag_file[:-4] + str(count).zfill(5) +".png"}
            write_to_csv(timePOS, csvPath)

            count += 1

            if(count == 1):
                firstTime = t
            lastTime = t



        totTime = lastTime - firstTime
        print("Total time:", totTime)
        print("Total time sec:", totTime/(10**9))
        print("Number of measurements:", count)
        print("Avg time:", totTime/count)
        print("Avg time sec:", (int(str(totTime))/count)/(10**9))
        print("First", firstTime)
        print("Last", lastTime)
        
        

    bag.close()
    

