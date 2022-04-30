import numpy as np
import cv2
import pandas as pd
import scipy.signal as sig
for j in range(168,175): # j is ring index
    file=pd.read_csv("F:\\summer project\\image processing\\Trajectories_only_crop_new\\newringcrp"+str(j)+".txt")
    print(j)
    for i in range(file.shape[0]-10):  # i is frame index in a particular ring(limited to frame_count-frame diff)
        img1 = cv2.imread("F:\\summer project\\image processing\\roi_extraction\\ring_"+str(j)+"\\"+"ring_"+str(j)+"frame_"+str(i)+".jpg")
        img1 = img1[8:16, 30:60]
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread("F:\\summer project\\image processing\\roi_extraction\\ring_"+str(j)+"\\"+"ring_"+str(j)+"frame_"+str(i+10)+".jpg")
        img2 = img2[8:16, 30:60]
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        a1 = img1[:, :, 2]
        b1 = img2[:, :, 2]
        a = a1 - np.mean(a1)
        b = b1 - np.mean(b1)
        cc = sig.correlate2d(b, a)
        cc = abs(cc)
        ravel_range = [i for i in range(len(cc.ravel()))]
        x = ravel_range
        y = list(cc.ravel())
        maxima = max((y))
        max_loc = y.index(maxima)
        max_points = []
        for i in range(max_loc - 20, max_loc + 20):
            if y[i] == maxima:
                max_points.append(i)
        if len(max_points) >= 2:
            print("detected_ring=",j)



