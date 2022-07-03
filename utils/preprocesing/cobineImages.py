import cv2
from os import listdir, rename, makedirs
from tqdm import tqdm
import numpy as np

loadPath = "../data/lidarImages16"
drivePaths = listdir(loadPath)
drivePaths.sort()

#print(drivePaths[:8])

totalMean = np.zeros(4)
totalStd = np.zeros(4)

count = 0

print(drivePaths[:8])

drivepath_Count = 0
for drivePath in drivePaths:
    drivepath_Count += 1
    fullDrivePath = loadPath + "/" + drivePath 
    signalPath = fullDrivePath + "/signal_image" #choose singal image because it has the same number of images as the other folders that are going to be combined

    imageNames = listdir(signalPath)
    imageNames.sort()

    print("Drive: ", drivePath)
    makedirs("../data/combinedImages/" + drivePath, exist_ok=True)
    makedirs("../data/combinedNumpy16/structuredInFolders/" + drivePath, exist_ok=True)
    makedirs("../data/combinedNumpy16/" + "all", exist_ok=True)

    for i, imageName in enumerate(tqdm((imageNames))):
        try:
            img1 = cv2.imread(fullDrivePath + "/signal_image/" + imageName, 0)
            if (drivePath == "plains_drive"):
                img2 = cv2.imread(fullDrivePath + "/reflec_image/" + imageNames[i-1], 0) #plains drive is missing one frame, so this syncs the frames 
            else:
                img2 = cv2.imread(fullDrivePath + "/reflec_image/" + imageName, 0)
            img4 = cv2.imread(fullDrivePath + "/nearir_image/" + imageName, 0)
            img3 = cv2.imread(fullDrivePath + "/rangeBrightWhole/" + imageName, -1)
            

        #print(img1.shape)
        #print(img2.shape)
        #print(img3.shape)
        #print(img4.shape)
            dim = (2048, 64*4)
            img1Taller = cv2.resize(img1, dim, interpolation = cv2.INTER_NEAREST)
            img2Taller = cv2.resize(img2, dim, interpolation = cv2.INTER_NEAREST)
            img3Taller = cv2.resize(img3, dim, interpolation = cv2.INTER_NEAREST)
            img4Taller = cv2.resize(img4, dim, interpolation = cv2.INTER_NEAREST)

            #img = cv2.merge((img1Taller, img2Taller, img3Taller))
            #cv2.imwrite("../data/combinedImages/" + drivePath + "/" + imageName, img)
            #cv2.imwrite("../data/combinedImages/" + "all" + "/" + imageName, img)


            stacked4channels = np.dstack((img1Taller, img2Taller, img3Taller, img4Taller))

            mean, std = cv2.meanStdDev(stacked4channels)
            #print(mean, std)
            mean = mean.reshape(4)
            std = std.reshape(4)
            #print(mean.shape)

            totalMean += mean
            totalStd += std
            count += 1
            #val = np.reshape(stacked4channels[:,:,0], -1)


            print(stacked4channels[200:250,:,2])
            #print(val)
            #np.save("../data/combinedNumpy16/structuredInFolders/" + drivePath + "/" + imageName[:-4] + ".npy", stacked4channels)
            #if (drivepath_Count <= 8):
            #    np.save("../data/combinedNumpy16/all/" + imageName[:-4] + ".npy", stacked4channels)
            #else:
            #    np.save("../data/combinedNumpy16/" + "all" + "/" + drivePath + imageName[5:-4] + ".npy", stacked4channels)
            #np.save("../data/combinedNumpy16/" + "all" + "/" + drivePath + imageName[:-4] + ".npy", stacked4channels)
            #print(drivePath + imageName[5:-4])
            #np.save("../data/combinedNumpy16/" + "all" + "/" + drivePath + imageName[5:-4] + ".npy", stacked4channels)
            break
        except:
            print("failed on: ", fullDrivePath, " Image number:", imageName)


        

totalMeanAvg = totalMean / count
totalStdAvg =  totalStd /count
print("total mean", totalMeanAvg)
print("total std", totalStdAvg)
        

        