import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy import interpolate
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
ravel_range=[i for i in range(len(cc.ravel()))]
#file=open("cc.txt","w")
#file.write(str(cc))
#file.close()
#maxval=np.max(abs(cc[:]))
x=ravel_range
y=list(cc.ravel())
#func=interpolate.interp1d(x,y)
#x1=ravel_range
#y1=func(x1)

#plt.plot(ravel_range,cc.ravel())
"""
maxima=max(y)
max_loc=y.index(maxima)
max_points=[]
for i in range(max_loc-20,max_loc+20):
    if y[i]==maxima:
        max_points.append(i)

print(np.mean(max_points))
print(max_loc)
"""
maxima=max(y)
max_loc=y.index(maxima)

xy_collect=[]
y_collect=[]
for i in range(max_loc-200,max_loc+200):
    if y[i]==maxima:
        xy_collect.append(i*y[i])
        y_collect.append(y[i])
xy_sum=sum(xy_collect)
y_sum=sum(y_collect)
centroid=xy_sum/y_sum
print(max_loc,centroid)
curve=np.polyfit(x,y,100) #return coeff of polynomial
poly=np.poly1d(curve)
"""
plt.plot(x,y)
plt.plot([centroid for i in x],[i for i in y],color="g")
plt.plot(x,poly(x),color="r")
"""
cv2.imshow("cc",cc)
cv2.waitKey()




"""
y_peak=int(maxloc[0]+1)
x_peak=int(maxloc[1]+1)
print(x_peak,y_peak)
corr_offset=[y_peak-b.shape[0],x_peak-b.shape[1]]
print(corr_offset)
"""
