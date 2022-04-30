import pandas as pd
for j in range(2643,3959):
 print(j)
 f1=open("F:\\summer project\\image processing\\cross correlation\\vel_data_frame"+"\\"+"diff_10_frame_"+str(j)+".txt","a")
 f1.write("frame,ring,x,y,b,a,phi,x_shift,vel_ring"+"\n")
 for i in range(175):
  file = pd.read_csv("F:\\summer project\\image processing\\Trajectories_only_crop_new" + "\\" + "newringcrp" + str(i) + ".txt")
  track_data = pd.read_csv("F:\\summer project\\image processing\\cross correlation\\data extraction\\frame_diff_10"+"\\"+"ring_"+str(i)+".txt")
  frames_count = file.shape[0]  # no. of frames data in a typical ring txt file
  frames_count_vel = track_data.shape[0]
  x=file["x"]
  y = file["y"]
  b=file["b"]
  a=file["a"]
  phi=file["phi"]
  x_shift=track_data["X_shift"]
  if j >= frames_count_vel:
   continue
  f1.write(str(j)+","+str(i)+","+str(x[j])+","+str(y[j])+","+str(b[j])+","+str(a[j])+","+str(phi[j])+","+str(x_shift[j])+","+str(track_data["vel_X"][j])+"\n")
