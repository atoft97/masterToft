
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
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
import numpy as np

input_toptic = "/ugv_sensors/lidar/image"
bag_files_folder = "/home/potetsos/lagrinatorn/master/bagfiler/newBags"
output_dir = "/home/potetsos/skule/2022/masterCode/masterToft/data/pointClouds"
output_dir_all = "/home/potetsos/skule/2022/masterCode/masterToft/data/pointCloudAll"
#print("Extract images from %s on topic %s into %s" % (args.bag_file, args.image_topic, args.output_dir))

bag_files = listdir(bag_files_folder)
#bag_file = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/stand_still_short.bag"
print(bag_files)
#print(fail)

import struct

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
    os.makedirs(output_dir_all, exist_ok=True)

    print("\n")

    for topicI in ['/ugv_sensors/lidar/cloud/points']:
        count =0
        for topic, msg, t in tqdm(bag.read_messages(topics=topicI)): #tqdm

            #print("Image topic: ", topic)
            #print(msg)

           # print(type(msg))
            #pc = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
            #print(pc)
            #int_data = list(pc)
            #print(int_data)
            #points=np.zeros((pc.shape[0],3))
            #points[:,0]=pc['x']
            #points[:,1]=pc['y']
            #points[:,2]=pc['z']

            #print(points)
            #pc_np = np.load("test.npy")
            pc_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)
            #print(type(pc_np))
            #print(len(np.unique(pc_np)))

            np.save(savePathBag + "/" + bag_file[:-4] + "_" + str(count).zfill(5) + ".npy", pc_np)
            np.save(output_dir_all + "/" + bag_file[:-4] + "_" + str(count).zfill(5) + ".npy", pc_np)
            count+=1

            




            


    bag.close()
    

