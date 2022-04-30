
import cv2
import numpy as np
import pandas as pd
import os
for j in range(22,23): #j can be thought as ring index
 file=pd.read_csv("F:\\summer project\\image processing\\Trajectories_only_crop_new"+"\\"+"newringcrp"+str(j)+".txt")
 track_data=pd.read_csv("F:\\summer project\\image processing\\cross correlation\\data extraction\\frame_diff_10"+"\\"+"ring_"+str(j)+".txt")
 if file.shape[0]>0:
     x_shift=track_data["X_shift"]
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
     cap = cv2.VideoCapture("F:\\summer project\\image processing\\Trajectories_only_crop_new\Dynamics.mov")
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
         if i%10==0 :
          t=x_shift[i]
          if t>0:
           cv2.arrowedLine(roi, (40, 9),(70,9),(0,0,255),2,cv2.LINE_4)
          elif t==0:
           pass
          else:
           cv2.arrowedLine(roi, (40, 9), (10, 9), (0, 255, 255 ), 2,cv2.LINE_8)
         cv2.imshow("roi_"+str(j), roi)

         #cv2.imwrite(r"F:\summer project\image processing\roi_extraction"+"\\"+"ring_"+str(j)+"\\"+"ring_"+str(j)+"frame_"+str(i)+".jpg",roi)
         #os.chdir(r"F:\summer project\image processing\Trajectories_only_crop_new")
         k = cv2.waitKey(4)
         i = i + 1
         if i==file.shape[0]:
          break
     cap.release()
     print(j)
     cv2.destroyAllWindows()
 else:
  continue


"""
import pandas as pd
import matplotlib.pyplot as plt
orig_data=pd.read_csv("F:\\summer project\\image processing\\Trajectories_only_crop_new\\newringcrp22.txt")
cc_data=pd.read_csv("F:\\summer project\\image processing\\cross correlation\\data extraction\\frame_diff_10\\ring_22.txt")
x=orig_data["x"].values
x_disp_data_orig=[]
frame_no=cc_data["frame_initial"].values
x_disp_data_cc=cc_data["X_shift"].values
for i in range(len(x)-10):
 shift=x[i+10]-x[i]
 x_disp_data_orig.append(shift)
plt.plot(frame_no,x_disp_data_orig,color="g",label="original shift")
plt.plot(frame_no,-1*x_disp_data_cc,color="c",label="cc data")
plt.legend()
plt.show()
"""