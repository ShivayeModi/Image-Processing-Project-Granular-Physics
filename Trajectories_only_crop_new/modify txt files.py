#HOW TO CREATE A NEW FOLDER OR MAKE A NEW DIRECTORY
import os
import cv2
img=cv2.imread("latestphoto.jpg")
a="C:\\Users\\shiva\\Desktop"
b="artificial folder"
os.chdir(a)
os.mkdir(b)
os.chdir(a+"\\"+ b)
os.mkdir("shivaye")
cv2.imwrite(a+"\\"+b+"\\"+"shivaye"+"\\"+img)

"""
#EDITING THE LARGE NUMBER OF FILES
for i in range(180):
 file=open("ringcrp"+str(i)+".txt","r")   #r stands for read mode
 new=open("newringcrp"+str(i)+".txt","w")  # w stands for write mode
 data=file.read()                 
 columns="frame_no,x,y,b,a,phi"
 new.write(columns+"\n"+str(data))
 file.close()
 new.close()
"""