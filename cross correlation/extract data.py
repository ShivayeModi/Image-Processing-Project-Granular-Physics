import cv2
import numpy as np
import pandas as pd
import os
from scipy import signal as sig
frame_diff=30
#os.chdir("F:\\summer project\\image processing\\cross correlation\\data extraction")
#os.mkdir("frame_diff_"+str(frame_diff))
for i in range(1,180): # i can be thought as a ring number
    file=pd.read_csv("F:\\summer project\\image processing\\Trajectories_only_crop_new"+"\\"+"newringcrp"+str(i)+".txt")
    ring_frames=file.shape[0]
    print(i)
    if ring_frames>0: #file.shape[0] indicates number of frames for a particular ring
        txtfile=open("F:\\summer project\\image processing\\cross correlation\\data extraction"+"\\"+"frame_diff_"+str(frame_diff)+"\\"+"ring_"+str(i)+".txt","a")
        txtfile.write("frame_initial," + "frame_final," + "X_shift," + "Y_shift," + "vel_X"+"\n")
        for j in range(0,file.shape[0]): #j can be thought as a frame number for the ith ring
            img1=cv2.imread("F:\\summer project\\image processing\\roi_extraction\\ring_"+str(i)+"\\"+"ring_"+str(i)+"frame_"+str(j)+".jpg")
            img1 = img1[8:16, 30:60]
            img2=cv2.imread("F:\\summer project\\image processing\\roi_extraction\\ring_"+str(i)+"\\"+"ring_"+str(i)+"frame_"+str(j+frame_diff)+".jpg")
            img2 = img2[8:16, 30:60]
            a1 = img1[:, :, 1]
            b1 = img2[:, :, 1]
            a = a1 - np.mean(a1)
            b = b1 - np.mean(b1)
            cc = sig.correlate2d(b, a)
            maxval = np.max(abs(cc[:]))
            maxloc = np.where(abs(cc) == maxval)
            y_peak = int(maxloc[0] + 1)
            x_peak = int(maxloc[1] + 1)
            corr_offset = [y_peak - b.shape[0], x_peak - b.shape[1]]
            y_shift=corr_offset[0]
            x_shift=corr_offset[1]
            vel_x=x_shift/frame_diff
            vel_x=round(vel_x,4)
            txtfile.write(str(j)+","+str(j+frame_diff)+","+str(x_shift)+","+str(y_shift)+","+str(vel_x)+"\n")
            if j==ring_frames-frame_diff-1:
                break
        txtfile.close()
    else:
     continue