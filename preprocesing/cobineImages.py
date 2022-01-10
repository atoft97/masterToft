import cv2
from os import listdir, rename, makedirs
from tqdm import tqdm


loadPath = "../data/lidarImages"
drivePaths = listdir(loadPath)

for drivePath in drivePaths:
    fullDrivePath = loadPath + "/" + drivePath 
    signalPath = fullDrivePath + "/signal_image" #choose singal image because it has the same number of images as the other folders that are going to be combined

    imageNames = listdir(signalPath)
    imageNames.sort()

    print("Drive: ", drivePath)
    makedirs("../data/combinedImages/" + drivePath, exist_ok=True)

    for i, imageName in enumerate(tqdm((imageNames))):
        #print(filename[:-3])
        #rename(filename, filename[:-3])

        #rename("data/Video"+ paddedNumber +"_Ano/images_int/" + filename, "data/Video"+ paddedNumber +"_Ano/images_int/" + filename[:-3])
        #rename("data/Video"+ paddedNumber +"_Ano/images_int/" + filename, "data/Video"+ paddedNumber +"_Ano/images_int/" + filename[:-3])
        #rename("data/Video"+ paddedNumber +"_Ano/images_range/" + filename, "data/Video"+ paddedNumber +"_Ano/images_range/" + filename[:-3])


        try:
            img1 = cv2.imread(fullDrivePath + "/signal_image/" + imageName, 0)
            if (drivePath == "plains_drive"):
                img2 = cv2.imread(fullDrivePath + "/reflec_image/" + imageNames[i-1], 0)
            else:
                img2 = cv2.imread(fullDrivePath + "/reflec_image/" + imageName, 0)
            img3 = cv2.imread(fullDrivePath + "/rangeBright/" + imageName, 0)

            img = cv2.merge((img1, img2, img3))

            cv2.imwrite("../data/combinedImages/" + drivePath + "/" + imageName, img)
        except:
            print("failed on: ", fullDrivePath, " Image number:", imageName)


        

        