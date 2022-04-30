import cv2
import numpy as np
import pandas as pd
import os
for j in range(69,70): #j can be thought as ring index
 file=pd.read_csv("newringcrp"+str(j)+".txt")
 if file.shape[0]>0:
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
     i = 0                    # i will be used as frame number of the running video
     cap = cv2.VideoCapture("Dynamics.mov")
     #os.chdir(r"F:\summer project\image processing\roi_extraction")
     #os.mkdir("ring_" + str(j))
     while (True):
         ret, frame = cap.read()
         pts = np.array(
             [(int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (int(x4[i]), int(y4[i])), (int(x3[i]), int(y3[i]))])
         cv2.polylines(frame, [pts], True, (255, 255, 255), 1)
         rotmatrix = cv2.getRotationMatrix2D((int(x0[i]), int(y0[i])), -1 * (90 + phid[i]), 1)
         rotrect = cv2.warpAffine(frame, rotmatrix, (frame.shape[1], frame.shape[0]))
         roi = rotrect[int(y0[i]) - int(0.5 * b):int(y0[i]) + int(0.5 * b),
               int(x0[i]) - int(0.5 * a):int(x0[i]) + int(0.5 * a)]
         cv2.imshow("frame"+str(j), frame)
         #cv2.imshow("rotrect"+str(j), rotrect)
         cv2.imshow("roi_"+str(j), roi)
         #cv2.imwrite(r"F:\summer project\image processing\roi_extraction"+"\\"+"ring_"+str(j)+"\\"+"ring_"+str(j)+"frame_"+str(i)+".jpg",roi)
         #os.chdir(r"F:\summer project\image processing\Trajectories_only_crop_new")
         k = cv2.waitKey(1)
         i = i + 1
         if i==file.shape[0]:
          break
     cap.release()
     print(j)
     cv2.destroyAllWindows()
 else:
  continue
