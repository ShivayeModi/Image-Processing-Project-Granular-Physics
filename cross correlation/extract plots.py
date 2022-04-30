import os
from matplotlib import pyplot as plt
import pandas as pd
frame_diff=30
os.chdir("F:\\summer project\\image processing\\cross correlation\\line plot extraction")
os.mkdir("X-shift-plot_frame_diff_"+str(frame_diff))
for j in range(1,175): #j indicates ring number
 print(j)
 file=pd.read_csv("F:\\summer project\\image processing\\cross correlation\\data extraction\\frame_diff_"+str(frame_diff)+"\\"+"ring_"+str(j)+".txt")
 X_shift=file["X_shift"]
 frame_initial=file["frame_initial"]
 vel_X=file["vel_X"]
 file.plot("frame_initial","X_shift",color="r")
 plt.xlabel("frame_initial")
 plt.ylabel("X_shift")
 plt.title("ring_"+str(j))
 plt.savefig("F:\summer project\image processing\cross correlation\line plot extraction"+"\\"+"X-shift-plot_frame_diff_"+str(frame_diff)+"\\"+"ring_"+str(j)+".png")
 plt.close("ring_"+str(j))