import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig
frame=[]
x_shift=[]
frame_diff=10
for j in range(0,3969): #j can be thought of frame index in a particular ring
 img1=cv2.imread("F:\\summer project\\image processing\\roi_extraction\\ring_22"+"\\"+"ring_22frame_"+str(j)+".jpg")
 img1=img1[8:16,30:60]
 img2=cv2.imread("F:\\summer project\\image processing\\roi_extraction\\ring_22"+"\\"+"ring_22frame_"+str(j+frame_diff)+".jpg")
 img2=img2[8:16,30:60]
 a1=img1[:,:,1]
 b1=img2[:,:,1]
 a=a1-np.mean(a1)
 b=b1-np.mean(b1)
 cc=sig.correlate2d(b,a)
 maxval=np.max(abs(cc[:]))
 maxloc=np.where(abs(cc)==maxval)
 y_peak=int(maxloc[0]+1)
 x_peak=int(maxloc[1]+1)
 corr_offset=[y_peak-b.shape[0],x_peak-b.shape[1]]
 frame.append(j)
 x_shift.append(corr_offset[1])
 print("frame_"+str(j),"frame_"+str(j+frame_diff),corr_offset)
 if j==3968-frame_diff:
  break
vel=[i/frame_diff for i in x_shift]
print(set(x_shift))
plt.plot(frame,x_shift)
plt.title("frame difference_"+str(frame_diff))
plt.xlabel("1st frame number")
plt.ylabel("X_shift")
plt.legend()
plt.show()