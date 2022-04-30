import pandas as pd
file=open("ringcrp18.txt","r")
new=open("newringcrp18.txt","w")
data=file.read()
columns="frame_no,x,y,b,a,phi"
new.write(columns+"\n"+str(data))

file.close()
new.close()
