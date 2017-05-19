print('Importing modules...' , end='\t')
import os
import numpy as np
from cv2 import imread 
import seaborn as sns
import matplotlib.pylab as plt
import scipy.stats as sp
print('Finished!')

def CreateIntensityProfile(height, width, strFolder):
    "Create 1D and 2D average intensity profile of images \
     return: (intensity_1D, intensity_2D, intensity_3D) "

    print('Reading imges from '+strFolder+'...' , end='')
    img_stack = np.zeros((height,width,100))

    # Read grayscale image and stack in 3D form
    for i, file in enumerate(os.listdir(strFolder)):img_stack[:,:,i]=imread(os.path.join(strFolder,file), 0)

    print('Finished!')
    return np.mean(img_stack, axis=(0,1)), np.mean(img_stack, axis=2), img_stack

def Outliers1D(minmax, sample_1D):
    idx = (sample_1D<minmax[0]*0.95) | (sample_1D>minmax[1]*1.05)
    return (np.where(idx), intensity1D[idx])

def Outliers2D(ref_2D, sample_3D, threshold):
    ref_3D = np.repeat(ref_2D[:,:,np.newaxis], sample_3D.shape[2], axis=2)
    RMS = np.sqrt(np.mean(np.square(ref_3D-sample_3D),axis=(0,1)))
    idx = RMS>threshold
    return (np.where(idx), RMS[idx])

height = 1728; width = 2336

# Create 2D & 1D average intensity data
curfolder = os.getcwd()
ref_path = os.path.join(curfolder,'ref')
ref_1D, ref_2D = CreateIntensityProfile(height, width, ref_path)[:2]

nobs, minmax, mean, variance = sp.describe(ref_1D)[:4]; std = np.sqrt(variance) 
print(nobs, minmax, mean, variance)
rv = sp.t(df=nobs-1, loc=mean, scale=std)

# KDE plot of the intensity distribution and its estimated t pdf
sns.distplot(ref_1D, kde=True, rug=True);
#xx = np.linspace(minmax[0],minmax[1],150)
#plt.plot(xx, rv.pdf(xx))
#plt.show()

# Find outliers in the samples
for folder in os.listdir(curfolder):
    full_dir = os.path.join(curfolder, folder)
    if folder != 'ref' and os.path.isdir(full_dir): 
        intensity1D, intensity2D, img_stack = CreateIntensityProfile(height, width, full_dir)
        print(sp.describe(intensity1D)[:4])
        outliers_1D = Outliers1D(minmax, intensity1D); print(outliers_1D)
        outliers_2D = Outliers2D(ref_2D, img_stack, 1); print(outliers_2D)  
        sns.distplot(intensity1D, kde=True, rug=True);
        
plt.show()        


'''
# Display sample code
cv2.normalize(img_stack[:,:,0], img_stack[:,:,0], 0 , 1)
cv2.imshow('image',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''