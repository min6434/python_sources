import os.path
import numpy as np
#import sklearn
import cv2

img_stack = np.zeros((1728,2336,100))
dir = '.\\image\\2017_05_15_16_42_44_'
ext = '.tif'

# Read grayscale image and stack in 3D form
for i in range(100): img_stack[:,:,i] = cv2.imread(dir+str(i+1).zfill(4)+ext, 0)

intensity = np.mean(img_stack, axis=(0,1))
#print(np.std(img_stack))

'''
# Display sample code
cv2.normalize(img_stack[:,:,0], img_stack[:,:,0], 0 , 1)
cv2.imshow('image',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''