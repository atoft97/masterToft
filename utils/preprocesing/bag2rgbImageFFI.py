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

#parser = argparse.ArgumentParser(description='Extract lidar images from a ROS-bags in a folder.')
#parser.add_argument("bag_files_folder", help="Input ROS bag (../data/bagFiles)")
#parser.add_argument("output_dir", help="Output directory (../data/lidarImages)")
#parser.add_argument("image_topic", help="Image topic (/ugv_sensors/lidar/image)")

#args = parser.parse_args()

#print("Extract images from %s on topic %s into %s" % (args.bag_file, args.image_topic, args.output_dir))
#bag_files_folder = "/home/potetsos/lagrinatorn/master/bagfiler/bagFiles"
bag_files_folder = "/home/potetsos/lagrinatorn/master/bagfiler/nyBag"
bag_files = listdir(bag_files_folder)
#bag_files = ["plains_drive.bag"]
#bag_file = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/stand_still_short.bag"

#bag2Name = {"plains_drive.bag": "plainsDrive", "plains_drive2.bag": "plainsDrive", "road_drive.bag": "roadDrive", "road_drive2.bag": "roadDrive2", "rock_quarry_drive.bag": "rockQuarryDrive", "rock_quarry_into_woods_drive.bag": "rockQuarryIntoWoodsDrive", "stand_still_short.bag": "stille"}
output_dir = "/home/potetsos/lagrinatorn/master/bilder/rgb"
bag_files = listdir(bag_files_folder)
print(bag_files)

for bag_file in bag_files[4:]:
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
    #print(fail)
    savePathBag = f"{output_dir}/{bag_file[:-4]}"
    os.makedirs(savePathBag, exist_ok=True)



    for imageToptic in ['/ugv_sensors/camera/color/image/compressed']:
    #for imageToptic in ['/ugv_sensors/lidar/image/signal_image']:
        
        #savePathTopic = savePathBag + "/" + imageToptic
        #os.makedirs(savePathTopic, exist_ok=True)


        print("Image topic: ", imageToptic)
        count = 0
        for topic, msg, t in tqdm(bag.read_messages(topics=imageToptic)): 

            #print(msg.header)
            #print(t)
            #print(msg)
            cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            #print(savePathBag + "/test" + str(count).zfill(5) +".png")
            cv2.imwrite(savePathBag + "/" + bag_file[:-4] + "_" + str(count).zfill(5) +".png", cv_img)
            
            #timePOS = {'time': t,'filename': bag_file[:-4] + "_frame" + str(count).zfill(5) +".png"}
            #timePOS = {'time': f"{msg.header.stamp.secs}{msg.header.stamp.nsecs}",'filename': bag2Name[bag_file] + "_frame" + str(count).zfill(5) +".png"}
            #timePOS = {'time': f"{msg.header.stamp.secs}{msg.header.stamp.nsecs}",'filename': bag_file[:-4] + str(count).zfill(5) +".png"}
            #print(timePOS)
            #print(f"{msg.header.stamp.secs}{msg.header.stamp.nsecs}")
            #print(t)

            #print("\n")
            
            #"/frame" + str(count).zfill(5) +".png"
            #write_to_csv(timePOS, csvPath)

            count += 1

            #if(count == 1):
            #    firstTime = t
            #lastTime = t

            #if (count == 4506):
            #    break   
            #if (count == 1927):
            #   break  
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
    
    
'''

1632987954
544527531

1632987954
648865634

1632987954
567358976

1632987954
688847341





1633008447
172354221
1633008449
508977033



1633008447
172354221
1633008449
505140761



1633008446
905359268
1633008449
277714098


1633008447
3068928

1633008447
128144679











RGB
1633008447
617345572

1633008449
991952805

LiDAR
1633008447
402590720

1633008447
528805238




RGB
1633008443
345440149
1633008445
301873739


LiDAR
1633008443
303786240

1633008443
436593135



RGB
1633008475
740718126

1633008477
870478433



LiDAR
1633008477
803036928

1633008477
926259853

'''
