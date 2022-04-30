import os
from matplotlib import pyplot as plt
import pandas as pd
frame_diff=30
os.chdir("F:\\summer project\\image processing\\cross correlation\\bar_plot_extraction2")
os.mkdir("X-shift-bar_frame_diff_"+str(frame_diff))

for j in range(1,175): #j indicates ring number
 print(j)
 file=pd.read_csv("F:\\summer project\\image processing\\cross correlation\\data extraction\\frame_diff_"+str(frame_diff)+"\\"+"ring_"+str(j)+".txt")
 X_shift=file["X_shift"]
 frame_initial=file["frame_initial"]
 vel_X=file["vel_X"]
 set_X_shift=set(X_shift)
 list_x_shift=[]
 list_x_shift_count=[]
 #print(set_X_shift)
 for i in range(len(set_X_shift)):
  minimum=min(set_X_shift)
  set_X_shift.remove(minimum)
  list_x_shift.append(minimum)
 #print(list_x_shift)
 for k in range(len(list_x_shift)):
  list_x_shift_count.append(len(X_shift[X_shift==list_x_shift[k]]))
 #print(list_x_shift_count)

 merged_data=pd.DataFrame({"count":list_x_shift_count},index=list_x_shift)
 merged_data.plot(kind="line", label="count",color="g")
 #merged_data.plot(kind="bar", label="count")

 plt.xlabel("X_shift")
 plt.ylabel("count")
 plt.title("ring_"+str(j))
 plt.legend()
 #plt.show()
 plt.savefig("F:\\summer project\\image processing\\cross correlation\\bar_plot_extraction2"+"\\"+"X-shift-bar_frame_diff_"+str(frame_diff)+"\\"+"ring_"+str(j)+".png")
 plt.close("ring_"+str(j))

