
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""
import matplotlib.pyplot as plt
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
from ouster import client
from ouster.client._client import ScanBatcher
#lidarScan = client.LidarScan()
#print(lidarScan)
import open3d as o3d

#from ouster_ros.msg import PacketMsg

#from ouster import client
#import struct

#from ouster_ros.msg import PacketMsg

metadata_path = "lidar_metadata.json"
with open(metadata_path, 'r') as f:
    info = client.SensorInfo(f.read())

#parser = argparse.ArgumentParser(description='Extract lidar images from a ROS-bags in a folder.')
#parser.add_argument("bag_files_folder", help="Input ROS bag (../data/bagFiles)")
#parser.add_argument("output_dir", help="Output directory (../data/lidarImages)")
#parser.add_argument("image_topic", help="Image topic (/ugv_sensors/lidar/image)")

#args = parser.parse_args()

#print("Extract images from %s on topic %s into %s" % (args.bag_file, args.image_topic, args.output_dir))

#bag_files = listdir(args.bag_files_folder)
#bag_file = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/stand_still_short.bag"
#print(bag_files)
#print(fail)


bag_file = "../../../bagFiles/stand_still_short.bag"
#bag_file = "../../../bagFiles/plains_drive.bag"
#for bag_file in bag_files:
print("\n")
print("Bag file: ", bag_file)

#bag = rosbag.Bag(args.bag_files_folder +"/"+ bag_file, "r")
bag = rosbag.Bag(bag_file, "r")
bridge = CvBridge()

topics = bag.get_type_and_topic_info()[1].keys()
#for topic in topics:
#    print(topic)

savePathBag = str("testLidarMsg")[:-4]
os.makedirs(savePathBag, exist_ok=True)

print("\n")



combinedBuffer = b''

ls = client.LidarScan(info.format.pixels_per_column,
                          info.format.columns_per_frame)
batch = ScanBatcher(info)
print(ls)

frame_id = 0

for topicI in ['/ugv_sensors/lidar/driver/lidar_packets']:
    count =0
    for topic, msg, t in tqdm(bag.read_messages(topics=topicI)): #tqdm

        #print("Image topic: ", topic)
        #print(type(msg))
        #print(msg.buf) 
        #print(type(msg.buf))

        #combinedBuffer += msg.buf

        #print(len(combinedBuffer))
        

        lidarMelding = client.LidarPacket(msg.buf, info)

        #print(lidarMelding._data)
        #batch(p._data, ls)

        packets_per_frame = 128

        batch(lidarMelding._data, ls)

        if (ls.frame_id != frame_id):
            frame_id = ls.frame_id
            ranges = ls.field(client.ChanField.RANGE)
            refect = ls.field(client.ChanField.REFLECTIVITY)
            signal = ls.field(client.ChanField.SIGNAL)

            print(signal)

            range_img = client.destagger(info, ranges)
            refect_img = client.destagger(info, refect)
            signal_img = client.destagger(info, signal)

            print(signal_img)

            print(ls.frame_id)
            cv2.imwrite(f"heimalageBilde/range/{frame_id-1}.png", range_img/255)
            cv2.imwrite(f"heimalageBilde/refect/{frame_id-1}.png", refect_img/1.0)
            cv2.imwrite(f"heimalageBilde/signal/{frame_id-1}.png", signal_img/1.0)
            #ls = client.LidarScan(info.format.pixels_per_column,
            #              info.format.columns_per_frame)

            xyzlut = client.XYZLut(info)
            xyz = xyzlut(ls.field(client.ChanField.RANGE))
            #xyz = client.XYZLut(metadata)(scan)

            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.reshape((-1, 3))))
            #axes = o3d.geometry.TriangleMesh.create_coordinate_frame(1.0)



        #print(ls.status)
        #batch = ScanBatcher(lidar_stream.metadata)

        #fields: Dict[client.ChanField, client.FieldDType] = {
        #    client.ChanField.RANGE: np.uint32,
        #    client.ChanField.SIGNAL: np.uint16
        #}
        #print(lidarMelding.frame_id)

        #ranges = ls.field(client.ChanField.RANGE)

        #print(ranges.shape)
        
        #range_img = client.destagger(info, ranges)

         #if frame_id change

        #print(range_img)
        #print(range_img.shape)
        #print(type(range_img))

        #cv_img = bridge.compressed_imgmsg_to_cv2(ranges, desired_encoding="bgr8")
        
        
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
        #pc_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=True)
        #print(type(pc_np))
        #print(len(np.unique(pc_np)))

        #np.save(savePathBag + "/" + bag_file[:-4] + str(count).zfill(5) + ".npy", pc_np)
        count+=1

        if (count == 10000):
            break



        


bag.close()
    
'''
0it [00:00, ?it/s]Image topic:  /tf_static
<class 'tmppmxf4cv0._tf2_msgs__TFMessage'>
transforms: 
  - 
    header: 
      seq: 0
      stamp: 
        secs: 1633005480
        nsecs:  56465556
      frame_id: "os_sensor"
    child_frame_id: "os_imu"
    transform: 
      translation: 
        x: 0.006253
        y: -0.011775
        z: 0.007645
      rotation: 
        x: 0.0
        y: 0.0
        z: 0.0
        w: 1.0
  - 
    header: 
      seq: 0
      stamp: 
        secs: 1633005480
        nsecs:  56496984
      frame_id: "os_sensor"
    child_frame_id: "os_lidar"
    transform: 
      translation: 
        x: 0.0
        y: 0.0
        z: 0.036180000000000004
      rotation: 
        x: 0.0
        y: 0.0
        z: 1.0
        w: 0.0
Image topic:  /tf_static
<class 'tmppmxf4cv0._tf2_msgs__TFMessage'>
transforms: 
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "E"
    child_frame_id: "Ep"
    transform: 
      translation: 
        x: 3021942.1862227945
        y: 609986.3036481969
        z: 5564863.606869828
      rotation: 
        x: 0.0962949136334806
        y: -0.963733768658767
        z: 0.024745173572206146
        w: 0.24765336488723916
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "tracked_tag_46"
    child_frame_id: "tracked_tag_46e"
    transform: 
      translation: 
        x: 0.0
        y: 0.0
        z: 0.0
      rotation: 
        x: 0.0
        y: 0.0
        z: 0.0
        w: 1.0
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "tracked_tag_48"
    child_frame_id: "tracked_tag_48e"
    transform: 
      translation: 
        x: 0.0
        y: 0.0
        z: 0.0
      rotation: 
        x: 0.0
        y: 0.0
        z: 0.0
        w: 1.0
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "Q"
    child_frame_id: "Qe"
    transform: 
      translation: 
        x: 0.0
        y: 0.0
        z: 0.0
      rotation: 
        x: 0.7071067811865476
        y: 0.7071067811865475
        z: 0.0
        w: 0.0
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "B"
    child_frame_id: "Be"
    transform: 
      translation: 
        x: 0.0
        y: 0.0
        z: 0.0
      rotation: 
        x: 1.0
        y: 0.0
        z: 0.0
        w: 0.0
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "B"
    child_frame_id: "Bcr"
    transform: 
      translation: 
        x: -0.85
        y: 0.0
        z: 1.3
      rotation: 
        x: 0.0
        y: 0.0
        z: 0.0
        w: 1.0
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "os1_sensor"
    child_frame_id: "os1_imu"
    transform: 
      translation: 
        x: 0.006253
        y: -0.011775
        z: 0.007645
      rotation: 
        x: 0.0
        y: 0.0
        z: 0.0
        w: 1.0
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "Bcr"
    child_frame_id: "Bcre"
    transform: 
      translation: 
        x: 0.0
        y: 0.0
        z: 0.0
      rotation: 
        x: 1.0
        y: 0.0
        z: 0.0
        w: 0.0
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "Cc"
    child_frame_id: "Cr"
    transform: 
      translation: 
        x: 0.31841060618700723
        y: -0.004266420355097711
        z: 0.007186344978823494
      rotation: 
        x: -0.001374795193503923
        y: -0.0019099850563762293
        z: -0.009913716065458492
        w: 0.9999480887171263
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "Cl"
    child_frame_id: "Lf"
    transform: 
      translation: 
        x: 0.295560970570829
        y: -0.127735977193847
        z: 0.012707060086336
      rotation: 
        x: -0.5518380347213453
        y: 0.5780086205482602
        z: 0.43560533633901705
        w: 0.4142810748269253
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "Lf"
    child_frame_id: "os1_lidar"
    transform: 
      translation: 
        x: 0.0
        y: 0.0
        z: 0.0
      rotation: 
        x: 1.0
        y: 0.0
        z: 0.0
        w: 0.0
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "tracked_tag_10"
    child_frame_id: "tracked_tag_10e"
    transform: 
      translation: 
        x: 0.0
        y: 0.0
        z: 0.0
      rotation: 
        x: 0.0
        y: 0.0
        z: 0.0
        w: 1.0
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "os1_lidar"
    child_frame_id: "os1_sensor"
    transform: 
      translation: 
        x: 0.0
        y: 0.0
        z: -0.03618
      rotation: 
        x: 0.0
        y: 0.0
        z: 1.0
        w: 0.0
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "Ep"
    child_frame_id: "Epe"
    transform: 
      translation: 
        x: 0.0
        y: 0.0
        z: 0.0
      rotation: 
        x: 0.7071067811865476
        y: 0.7071067811865475
        z: 0.0
        w: 0.0
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "Cc"
    child_frame_id: "Cl"
    transform: 
      translation: 
        x: -0.31973265278220364
        y: 0.004630405219092533
        z: 0.006170186618211005
      rotation: 
        x: -0.00022738717113409492
        y: 0.0069318813657622685
        z: -0.010216630353989785
        w: 0.9999237559834326
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "B"
    child_frame_id: "Cc"
    transform: 
      translation: 
        x: -6.602308242581456e-05
        y: -0.0015719774558890787
        z: -0.16999271904358365
      rotation: 
        x: 0.49805253028825286
        y: 0.5021282666973006
        z: 0.4973129939664886
        w: 0.502484494177801
  - 
    header: 
      seq: 0
      stamp: 
        secs: 0
        nsecs:         0
      frame_id: "N"
    child_frame_id: "Ne"
    transform: 
      translation: 
        x: 0.0
        y: 0.0
        z: 0.0
      rotation: 
        x: 0.7071067811865476
        y: 0.7071067811865475
        z: 0.0
        w: 0.0
2it [00:00, 99.15it/s]

'''




