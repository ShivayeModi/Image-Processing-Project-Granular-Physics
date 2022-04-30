import cv2
import numpy as np
import pandas as pd
import os


#os.chdir("F:\\summer project\\image processing\\cross correlation")
#os.mkdir("piv_set_diff_10_framescale")
for j in range(2625,3958): # j as frame index
 vel_data=pd.read_csv("F:\\summer project\\image processing\\cross correlation\\vel_data_frame"+"\\"+"diff_10_frame_"+str(j)+".txt")
 vel_ring=vel_data["vel_ring"]
 v_max=max(abs(vel_ring))
 img=cv2.imread("F:\\summer project\\image processing\\Frame operations"+"\\"+"frame"+str(j)+".jpg")
 for i in range(1,175): # i as ring index
    file = pd.read_csv("F:\\summer project\\image processing\\Trajectories_only_crop_new"+"\\"+"newringcrp"+str(i)+".txt")
    track_data = pd.read_csv("F:\\summer project\\image processing\\cross correlation\\data extraction\\frame_diff_10"+"\\"+"ring_"+str(i)+".txt")
    frames_count=file.shape[0] # no. of frames data in a typical ring txt file
    frames_count_vel=track_data.shape[0]
    if j>=frames_count or j>=frames_count_vel:
        continue
    vel_x = track_data["vel_X"]
    x0 = file["x"]
    y0 = file["y"]
    a = 80
    b = 18
    phi = ((np.pi) * (-1) * file["phi"]) / 180
    phid = (-1) * file["phi"]
    x1 = x0 + (0.5 * a * (np.sin(phi))) - (0.5 * b * (np.cos(phi)))
    y1 = y0 + (0.5 * a * (np.cos(phi))) + (0.5 * b * (np.sin(phi)))
    x2 = x0 + (0.5 * a * (np.sin(phi))) + (0.5 * b * (np.cos(phi)))
    y2 = y0 + (0.5 * a * (np.cos(phi))) - (0.5 * b * (np.sin(phi)))
    x4 = x0 - (0.5 * a * (np.sin(phi))) + (0.5 * b * (np.cos(phi)))
    y4 = y0 - (0.5 * a * (np.cos(phi))) - (0.5 * b * (np.sin(phi)))
    x3 = x0 - (0.5 * a * (np.sin(phi))) - (0.5 * b * (np.cos(phi)))
    y3 = y0 - (0.5 * a * (np.cos(phi))) + (0.5 * b * (np.sin(phi)))
    pts = np.array([(int(x1[j]), int(y1[j])), (int(x2[j]), int(y2[j])), (int(x4[j]), int(y4[j])), (int(x3[j]), int(y3[j]))])
    arrow_length = (40 /v_max) * vel_x[j]
    xnew = x0[j] + int(arrow_length) * np.cos(np.pi / 2 + phi[j])  # dont go in normal cartesian frame,think about cv window frame
    ynew = y0[j] - int(arrow_length) * np.sin(np.pi / 2 + phi[j])
    if vel_x[j] == 0:
        cv2.circle(img, (int(x0[j]), int(y0[j])), 3, (255, 0, 255), -1)
    elif abs(vel_x[j])==v_max:
        cv2.arrowedLine(img, (int(x0[j]), int(y0[j])), (int(xnew), int(ynew)), (0, 255, 120), 3, tipLength=2)
    else:
        cv2.arrowedLine(img, (int(x0[j]), int(y0[j])), (int(xnew), int(ynew)), (0, 255, 255),2,tipLength=1)
 print(j)
 cv2.imwrite("F:\\summer project\\image processing\\cross correlation\\piv_set_diff_10_framescale"+"\\"+"frame_"+str(j)+"diff_10"+".jpg",img)