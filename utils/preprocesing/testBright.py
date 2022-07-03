import cv2
import numpy as np
'''
bit16 = np.array([[0,1,2,3],
				[4,6,7,8],
				[9,10,11,12],
				[256,1000,10000,16000]])
print(bit16)

newImge = cv2.multiply(bit16, 10)

print(newImge)


cv2.imwrite("testBright.png", newImge)

lest = cv2.imread("testBright.png", -1)
print(lest)
'''



originalImage = cv2.imread("/home/potetsos/skule/2022/masterCode/masterToft/data/lidarImages16/stand_still_short/range_image/frame00001.png", -1)
#print(originalImage)
#print(originalImage.shape)
print(originalImage)
newImge = cv2.multiply(originalImage, 10) #multply by 15
#newImge[newImge <= 0] = 255*255 #caps at 255
#newImge = cv2.bitwise_not(newImge)

cv2.imwrite("testBright.png", newImge)

lest = cv2.imread("testBright.png", 0)
print(lest)

cv2.imwrite("testBright8bit.png", lest)

lest8b = cv2.imread("testBright8bit.png", 0)
print(lest8b)

numpyBilde = np.load("/home/potetsos/skule/2022/masterCode/masterToft/data/combinedNumpy16/all/stand_still_short00001.npy")
print(numpyBilde[:,:,2])

cv2.imwrite("fraNumpy.png", numpyBilde[:,:,2])

numpyRangeBilde = numpyBilde[:,:,2]

print(numpyRangeBilde[100:150, :])