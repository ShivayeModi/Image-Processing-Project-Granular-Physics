import pandas as pd

txtfile = open("F:\\summer project\\image processing\\cross correlation\\describe_data\\" + "describe_data.txt", "a")
txtfile.write("frame,count,mean,std,min,25%,50%,75%,max" + "\n")

for i in range(3959): #i as frame index
    file=pd.read_csv("F:\\summer project\\image processing\\cross correlation\\vel_data_frame"+"\\"+"diff_10_frame_"+str(i)+".txt")
    vel=file["vel_ring"]
    data=vel.describe()
    line=str(i)+","
    heading=["count","mean","std","min","25%","50%","75%","max"]
    for j in range(len(heading)):
     line=line+str(data[heading[j]].round(4))
     if j==len(heading)-1:
        break
     else:
      line=line+","
    txtfile.write(line+"\n")
