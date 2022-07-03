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
from ouster import client, pcap
from more_itertools import nth

pcap_path = "../../../tmpFiler/OS1-128_Rev-05_Urban-Drive.pcap"

lidarPacet = client.LidarPacket
print(lidarPacet)

metadata_path = "lidar_metadata.json"
with open(metadata_path, 'r') as f:
    metadata = client.SensorInfo(f.read())

source = pcap.Pcap(pcap_path, metadata)

#scans = iter(client.Scans(source))

#scan = next(scans)
#scans = client.Scans(source)
#iteratro = scans.__iter__()
#print(iteratro)
#scan = next(iteratro)
#scan = nth(scans, 2)

#print(scan)


counter = 0
for packet in source:
	if isinstance(packet, client.LidarPacket):
		print(packet)
	#counter +=1
	#if (counter == 50000):
#		break
