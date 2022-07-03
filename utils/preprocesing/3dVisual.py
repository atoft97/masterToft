import open3d as o3d
import numpy as np
import cv2
from ouster import client
import os
from tqdm import tqdm
#import open3d_tutorial as o3dtut

metadataPath = "lidar_metadata.json"
with open(metadataPath, "r") as f:
    metadataLidar = client.SensorInfo(f.read())


#startPath = "/home/potetsos/skule/2022/rapport/results/exp3/pred"
#startPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdatasetWithRGBCounter/test/lidarLabelColorProjected"
startPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdatasetWithRGBCounter/test/rgbProjected"
#startPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdatasetWithRGBCounter/test/lidarImageProjected"

pointPath = "/home/potetsos/lagrinatorn/master/ffiLiDARdatasetWithRGBCounter/test/lidarPointCloud"

#outputPath = "/home/potetsos/skule/2022/rapport/results/exp3/lidar3d"
outputPath = "/home/potetsos/skule/2022/rapport/results/exp3/rgb3d"
os.makedirs(outputPath, exist_ok=True)

for filename in tqdm(os.listdir(startPath)):

    #if (filename != "75.png"):
    #    continue

    pointFullPath = f"{pointPath}/{filename[:-4]}.npy"
    colorPath = f"{startPath}/{filename}"
    pointData = np.load(pointFullPath)

    colorImage = cv2.imread(colorPath)



    
    dim = (2048, 64)
    colorSmaller = cv2.resize(colorImage, dim, interpolation = cv2.INTER_NEAREST)
    colorSmaller = cv2.cvtColor(colorSmaller.astype("float32"), cv2.COLOR_BGR2RGB)
    colorStaggered = client.destagger(metadataLidar, colorSmaller, inverse=True)
    colorReshaped = (colorStaggered.astype(float) / 255).reshape(-1, 3)
    print(colorSmaller.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointData)
    pcd.colors = o3d.utility.Vector3dVector(colorReshaped)

    #o3d.visualization.draw_geometries([pcd])
    
    '''
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.1)
    o3d.visualization.ViewControl.set_lookat(vis.get_view_control(), [0, 0, 0])
    #o3d.visualization.ViewControl.set_up(vis.get_view_control(), [1, -10, 10])
    o3d.visualization.ViewControl.set_front(vis.get_view_control(), [2, 0,0])
    #ctr = vis.get_view_control()

    #ctr = vis.get_view_control()
    #ctr.rotate(10.0, 90)

    #vis = o3d.visualization.Visualizer()
    #vis.create_window()
    #vis.add_geometry(pcd)
    #ctr = vis.get_view_control()
    #print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    #ctr.change_field_of_view(step=40)
    #ctr.rotate(90.0, 90.0)
    #ctr.zoom(0.5)
    #print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    #vis.run()

    image = vis.capture_screen_float_buffer()
    #print(image.shape)
    
    o3d.io.write_image("3dBilde.png", image)


    vis.run()
    '''
    '''
    render = o3d.visualization.Visualizer()
    render.create_window()
    #model, mat = getModel()
    render.add_geometry(pcd)

    
    render.setup_camera(parameters.intrinsic, parameters.extrinsic)

    render.run()
    '''
    
    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2022-06-10-18-00-22.json")
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    print(pcd.get_center())
    
    o3d.visualization.ViewControl.set_lookat(vis.get_view_control(), pcd.get_center())
    o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.03)
    o3d.visualization.ViewControl.set_front(vis.get_view_control(), [-20,-1 ,3])
    o3d.visualization.ViewControl.set_up(vis.get_view_control(), [1, 0, 10])
    #vis.get_view_control().convert_from_pinhole_camera_parameters(parameters)
    vis.poll_events()
    vis.update_renderer()
    #vis.setup_camera(parameters.intrinsic, parameters.extrinsic)
    image = vis.capture_screen_image(f"{outputPath}/{filename}", do_render=True)

    #vis.run()
    vis.destroy_window()

    
    
    #[pcd], zoom=0.3412,
    #                              front=[0.4257, -0.2125, -0.8795],
    #                              lookat=[2.6172, 2.0475, 1.532],
    #                              up=[-0.0694, -0.9768, 0.2024]

    