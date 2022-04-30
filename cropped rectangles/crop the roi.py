#CROP RECT WITH NO BACKGROUND

import cv2
import numpy as np
from pandas import *
import matplotlib.pyplot as plt
file=read_excel(r"F:\summer project\image processing\coordinates data.xlsx") #read excel file
x0=file["X"] # x coordinate column of center of ring
y0=file["Y"] #y coordinate column of center of ring
a=int(file["a"].mean()) # integer value of mean of major axis column
b=int(file["b"].mean()) # integer value of mean of minor axis column
phid=(-1)*file["phi"]  #angle in degrees, negative value is due to cv window y-axis inversion
phi=((np.pi)*(-1)*file["phi"])/180 # phi is taken in radians,negative value is due to cv window y-axis inversion
# calculation of the vertices of the rectangle in conventional cartesian coordinate system
x1=x0+(0.5*a*(np.sin(phi)))-(0.5*b*(np.cos(phi)))
y1=y0+(0.5*a*(np.cos(phi)))+(0.5*b*(np.sin(phi)))
x2=x0+(0.5*a*(np.sin(phi)))+(0.5*b*(np.cos(phi)))
y2=y0+(0.5*a*(np.cos(phi)))-(0.5*b*(np.sin(phi)))
x4=x0-(0.5*a*(np.sin(phi)))+(0.5*b*(np.cos(phi)))
y4=y0-(0.5*a*(np.cos(phi)))-(0.5*b*(np.sin(phi)))
x3=x0-(0.5*a*(np.sin(phi)))-(0.5*b*(np.cos(phi)))
y3=y0-(0.5*a*(np.cos(phi)))+(0.5*b*(np.sin(phi)))
#iteration begins from number range of zero to the last value of column x1
for i in range(len(x1)):
 cv2.namedWindow("rect"+str(i),cv2.WINDOW_NORMAL) #empty window fit to computer screen with the desired name
 cv2.namedWindow("rotrect"+str(i), cv2.WINDOW_NORMAL) #empty window fit to computer screen with the desired name
 img=cv2.imread("rect"+str(i)+".jpg") #read all previous extracted rectangles , one at each iteration
 #create a 2D rotational matrix operator which rotates the image about a fixed point and at a certain angle
 #1st attribute is the point about which rotation is done,2nd attribute is the rotational angle(take care of the sign)
 #3rd attribute is the scale of the image to which rotation has to be applied ,set it as one
 rotmatrix=cv2.getRotationMatrix2D((int(x0[i]),int(y0[i])),-1*int(90+phid[i]),scale=1)
 #Affline transformation ensures that parallel lines in original image will remain parallel in rotated image
 #1st attribute is the image on which rotation matrix is applied,2nd attribute is the rotation matrix
 #3rd attribute is the size of the rotated image which is to be shown
 rotimg=cv2.warpAffine(img,rotmatrix,(img.shape[1],img.shape[0]))
 cv2.imshow("rect"+str(i),img) #show the original rectangular image with white background
 cv2.imshow("rotrect"+str(i),rotimg) #show the transformed horizontally inclined image
 cv2.imwrite("rotrect"+str(i)+".jpg",rotimg) #write the rotated rectangle image as jpg file
 #center of the blue rectangle doesn't move at all,select region of interest (roi) of area a*b
 roi=rotimg[int(y0[i])-int(0.5*b):int(y0[i])+int(0.5*b),int(x0[i])-int(0.5*a):int(x0[i])+int(0.5*a)]
 cv2.imshow("roi"+str(i),roi) #show the roi
 cv2.imwrite("roi"+str(i)+".jpg",roi) #write the roi image as jpg file
 cv2.waitKey(9000) # time delay of 9 seconds
 cv2.destroyAllWindows() #destroy all existing windows
