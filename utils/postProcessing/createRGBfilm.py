from tqdm import tqdm
import cv2
from os import listdir
import numpy as np

for imageFolder in listdir("/media/potetsos/TOSHIBA EXT/ntnu/lidar"):
    try:

        fullOutputPath = f"/home/potetsos/lagrinatorn/master/videos/lidar/{imageFolder}.mp4"
        fullInputPath = f"/media/potetsos/TOSHIBA EXT/ntnu/lidar/{imageFolder}/all"

        #output_fname = "/home/potetsos/lagrinatorn/master/videos/2022-04-08-13-49-07_2.mp4"
        output_fname = fullOutputPath
        frames_per_second = 15*2
        codec, file_ext = ("mp4v", ".mp4")

        output_file = cv2.VideoWriter(
                        filename=output_fname,
                        # some installation of opencv may not support x264 (due to its license),
                        # you can try other format (e.g. MPEG)
                        fourcc=cv2.VideoWriter_fourcc(*codec),
                        fps=float(frames_per_second),
                        #frameSize=(3840, 1080),
                        #frameSize=(2048, 3072),
                        frameSize=(2048, 512),
                        #frameSize=(2048, 540),
                        isColor=True,
                )

        #output_file = cv2.VideoWriter(output_fname, 0, 1, (4096,1536))


        print(output_fname)
        #print(output_file)
        #startPath = "/home/potetsos/lagrinatorn/master/segmented/2022-04-08-13-49-07"
        startPath = fullInputPath
        files = listdir(startPath)
        files.sort()
        #print(files)

        index = 0
        for filename in tqdm(files):
            #print(index)
            #print(filename)
            #frame = read_image(startPath + "/" +filename, 'RGB')
            frame = cv2.imread(startPath + "/" +filename)
            #print(frame.shape)
            if (frame.shape[0] == 3072):
                newImgage = np.zeros((4096, 1536, 3))
                inputImage = frame[:1536, :, :]
                segmentedImge = frame[1536:, :, :]

                #print(inputImage.shape)
                #print(segmentedImge.shape)

                frame = np.hstack((inputImage, segmentedImge))

                #newImgage[:2048, :] = inputImage
                #newImgage[2048:, :] = segmentedImge
                #print(frame.shape)
            #print(frame.shape)
            #print(frame.shape)
            output_file.write(frame)
            index +=1
            #cv2.imwrite(filename, frame)
            #video.write(cv2.imread(os.path.join(image_folder, image)))
    except:
        continue
        


'''
for imageFolder in listdir("/home/potetsos/lagrinatorn/master/segmented"):
    fullOutputPath = f"/home/potetsos/lagrinatorn/master/videos/{imageFolder}.mp4"
    fullInputPath = f"/home/potetsos/lagrinatorn/master/segmented/{imageFolder}"

    #output_fname = "/home/potetsos/lagrinatorn/master/videos/2022-04-08-13-49-07_2.mp4"
    output_fname = fullOutputPath
    frames_per_second = 15
    codec, file_ext = ("mp4v", ".mp4")

    output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    #frameSize=(3840, 1080),
                    #frameSize=(2048, 3072),
                    frameSize=(4096, 1536),
                    #frameSize=(2048, 540),
                    isColor=True,
            )

    #output_file = cv2.VideoWriter(output_fname, 0, 1, (4096,1536))


    print(output_fname)
    #print(output_file)
    #startPath = "/home/potetsos/lagrinatorn/master/segmented/2022-04-08-13-49-07"
    startPath = fullInputPath
    files = listdir(startPath)
    files.sort()
    #print(files)

    index = 0
    for filename in tqdm(files):
        #print(index)
        #print(filename)
        #frame = read_image(startPath + "/" +filename, 'RGB')
        frame = cv2.imread(startPath + "/" +filename)
        #print(frame.shape)
        if (frame.shape[0] == 3072):
            newImgage = np.zeros((4096, 1536, 3))
            inputImage = frame[:1536, :, :]
            segmentedImge = frame[1536:, :, :]

            #print(inputImage.shape)
            #print(segmentedImge.shape)

            frame = np.hstack((inputImage, segmentedImge))

            #newImgage[:2048, :] = inputImage
            #newImgage[2048:, :] = segmentedImge
            #print(frame.shape)
        #print(frame.shape)
        output_file.write(frame)
        index +=1
        #cv2.imwrite(filename, frame)
        #video.write(cv2.imread(os.path.join(image_folder, image)))

    






    from tqdm import tqdm
import cv2
from os import listdir


for imageFolder in listdir("/home/potetsos/lagrinatorn/master/segmented"):
    fullOutputPath = f"/home/potetsos/lagrinatorn/master/videos/{imageFolder}"
    fullInputPath = f"/home/potetsos/lagrinatorn/master/segmented/{imageFolder}"

    output_fname = "/home/potetsos/lagrinatorn/master/videos/2022-04-08-13-49-07_2.mp4"
    output_fname = fullOutputPath
    frames_per_second = 15
    codec, file_ext = ("mp4v", ".mp4")

    output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    #frameSize=(3840, 1080),
                    frameSize=(4096, 3072),
                    #frameSize=(1536, 4096),
                    #frameSize=(2048, 540),
                    isColor=True,
            )

    #output_file = cv2.VideoWriter(output_fname, 0, 1, (4096,1536))



    print(imageFolder)
    print(output_fname)
    startPath = "/home/potetsos/lagrinatorn/master/segmented/2022-04-08-13-49-07"
    startPath = fullInputPath 
    files = listdir(startPath)
    files.sort()
    #print(files)

    index = 0
    for filename in tqdm(files):
        #print(index)
        #print(filename)
        #frame = read_image(startPath + "/" +filename, 'RGB')
        frame = cv2.imread(startPath + "/" +filename)
        print(frame.shape)
        #print(frame.shape)
        output_file.write(frame)
        index +=1
        #cv2.imwrite(filename, frame)
        #video.write(cv2.imread(os.path.join(image_folder, image)))

    break
'''