import cv2
import numpy as np
import pandas as pd
import os
# GET THE MAXIMUM VELOCITY OF THE FASTEST RING
v_max_collect=[]
for k in range(175): #k is ring index
    file=pd.read_csv("F:\\summer project\\image processing\\cross correlation\\data extraction\\frame_diff_10"+"\\"+"ring_"+str(k)+".txt")
    vel_max=max(abs(file["vel_X"]))
    v_max_collect.append(vel_max)
max_vel=max(v_max_collect)
print(max_vel)
#os.chdir("F:\\summer project\\image processing\\cross correlation")
#os.mkdir("piv_set_diff_10_finalscale")
for j in range(1501,3959): # j as frame index
 img=cv2.imread("F:\\summer project\\image processing\\Frame operations"+"\\"+"frame"+str(j)+".jpg")
 for i in range(0,175): # i as ring index
    file = pd.read_csv("F:\\summer project\\image processing\\Trajectories_only_crop_new"+"\\"+"newringcrp"+str(i)+".txt")
    track_data = pd.read_csv("F:\\summer project\\image processing\\cross correlation\\data extraction\\frame_diff_10"+"\\"+"ring_"+str(i)+".txt")
    frames_count=file.shape[0] # no. of frames data in a typical ring txt file
    frames_count_vel=track_data.shape[0]
    if j>=frames_count or j>=frames_count_vel:
        continue
    vel_x = track_data["vel_X"]
    v_max = max_vel
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
    arrow_length = int((80 /max_vel) * vel_x[j])
    if vel_x[j] == 0:
        cv2.circle(img, (int(x0[j]), int(y0[j])), 3, (255, 0, 255), -1)
    elif vel_x[j]<0:
        xnew = int(0.5 * (x4[j] + x3[j])) + arrow_length * np.cos(np.pi / 2 + phi[j])  # dont go in normal cartesian frame,think about cv window frame
        ynew = int(0.5 * (y4[j] + y3[j])) - arrow_length * np.sin(np.pi / 2 + phi[j])
        cv2.arrowedLine(img, (int(0.5*(x4[j]+x3[j])), int(0.5*(y4[j]+y3[j]))), (int(xnew), int(ynew)), (0, 255, 255),2,tipLength=2)
    elif vel_x[j]>0:
        xnew = int(0.5 * (x1[j] + x2[j])) + arrow_length * np.cos(np.pi / 2 + phi[j])  # dont go in normal cartesian frame,think about cv window frame
        ynew = int(0.5 * (y1[j] + y2[j])) - arrow_length * np.sin(np.pi / 2 + phi[j])
        cv2.arrowedLine(img, (int(0.5*(x1[j]+x2[j])), int(0.5*(y1[j]+y2[j]))), (int(xnew), int(ynew)), (0, 255, 255),2,tipLength=2)
 print(j)
 cv2.imwrite("F:\\summer project\\image processing\\cross correlation\\piv_set_diff_10_finalscale"+"\\"+"frame_"+str(j)+"diff_10"+".jpg",img)
