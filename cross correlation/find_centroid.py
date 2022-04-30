import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal as sig
img1=cv2.imread("ring_20frame_28.jpg")
img1=img1[8:16,30:60]
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2=cv2.imread("ring_20frame_38.jpg")
img2=img2[8:16,30:60]
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
a1=img1[:,:,2]
b1=img2[:,:,2]
a=a1-np.mean(a1)
b=b1-np.mean(b1)
cc=sig.correlate2d(b,a)
maxval=np.max(abs(cc[:]))
maxloc=np.where(abs(cc)==maxval)
y_peak=int(maxloc[0]+1)    #row number
x_peak=int(maxloc[1]+1)    #column number

box=cc[maxloc[0]]

"""
cc_x_sum=[]
cc_sum=[]
for x in range(x_max-3,x_max+3):
    cc_x_sum.append(x*cc[x])
    cc_sum.append(cc[x])
centroid_x=sum(cc_x_sum)/sum(cc_sum)
print(centroid_x)
"""
print(x_peak,y_peak)
corr_offset=[y_peak-b.shape[0],x_peak-b.shape[1]]
print(corr_offset)
images=[img1,img2,cc]
titles=["img1","img2","correlation2d"]
for i in range(3):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
plt.show()