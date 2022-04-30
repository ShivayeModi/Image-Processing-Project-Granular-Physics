import pandas as pd

for i in range(175):
 print(i)
 file=pd.read_csv("F:\\summer project\\image processing\\cross correlation\\data extraction\\frame_diff_10\\ring_"+str(i)+".txt")
 txt=open("F:\\summer project\\image processing\\cross correlation\\variaton_matrix\\"+"var_matrix_ring_"+str(i)+".txt","w")
 txt.write(str(file.corr().round(3)))
 txt.close()