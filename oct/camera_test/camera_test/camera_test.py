print('Importing modules...' , end='\t')
import os
import numpy as np
from cv2 import imread 
import seaborn as sns
import matplotlib.pylab as plt
import scipy.stats as sp
print('Finished!')

def CreateIntensityProfile(height, width, strFolder):
    "Create 1D and 2D average intensity profile of images"
    print('Reading imges from '+strFolder+'...' , end='')
    img_stack = np.zeros((height,width,100))

    # Read grayscale image and stack in 3D form
    for i, file in enumerate(os.listdir(strFolder)):
        img_stack[:,:,i]=imread(os.path.join(strFolder,file), 0)

    intensity1D = np.mean(img_stack, axis=(0,1))
    intensity2D = np.mean(img_stack, axis=2)
    print('Finished!')
    return intensity1D, intensity2D

height = 1728; width = 2336
img_stack = np.zeros((height,width,100))
curfolder = os.getcwd()

# Create 2D & 1D average intensity data
ref_path = os.path.join(curfolder,'ref')
ref_1D, ref_2D = CreateIntensityProfile(height, width, ref_path)

nobs, minmax, mean, variance = sp.describe(ref_1D)[:4]; std = np.sqrt(variance) 
rv = sp.t(df=nobs-1, loc=mean, scale=std)

# KDE plot of the intensity distribution and its estimated t pdf
xx = np.linspace(minmax[0],minmax[1],150)
sns.distplot(ref_1D, kde=True, rug=True);
plt.plot(xx, rv.pdf(xx))
plt.show()

for folder in os.listdir(curfolder):
    full_dir = os.path.join(curfolder, folder)
    if folder != 'ref' and os.path.isdir(full_dir): 
        intensity1D, intensity2D = CreateIntensityProfile(height, width, full_dir)
            
'''
IntOutliers = 0
for sample in intensity1D:
    if rv.pdf(sample) < 0.99:
        IntOutliers += 1

print('Intensity outliers: ', IntOutliers)

MSEOutliers = 0
for index in range(nobs):
    print(np.mean(np.sqrt(np.square(intensity2D-img_stack[:,:,index]))))
    if np.mean(np.sqrt(np.square(intensity2D-img_stack[:,:,index]))) > 0.01:
        MSEOutliers += 1

print('Mean square error outliers: ', MSEOutliers)
'''

'''
# Display sample code
cv2.normalize(img_stack[:,:,0], img_stack[:,:,0], 0 , 1)
cv2.imshow('image',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''