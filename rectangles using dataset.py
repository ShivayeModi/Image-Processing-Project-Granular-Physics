#Identifying rectangles

import cv2
import random as r
from matplotlib import pyplot as plt
from pandas import *  # import all functions from pandas library
import numpy as np
cv2.namedWindow("img",cv2.WINDOW_NORMAL)   #creates an empty cv window with title "img" fit to computer screen
img=cv2.imread("3500.jpg")      #read the image
file=read_excel(r"F:\summer project\image processing\coordinates data.xlsx")  #read the excel file of coordinates data
x0=file["X"]   # X column is specified as x0 which is the x coordinate for center of our rectangle
y0=file["Y"]   # Y column is specified as y0 which is the y coordinate for center of our rectangle
a=file["a"].mean() # for major axis mean value of column "a"  is taken
b=file["b"].mean() # for minor axis mean value of column "b"  is taken
phi=((np.pi)*(-1)*file["phi"])/180 # phi is taken in radians,negative value is due to cv window y-axis inversion
# calculation of the vertices of the rectangle in conventional cartesian coordinate system
#order is same as that of calculating in paper
# for more information see photos of paper calculation
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
  #declaration of array of four vertices
  pts=np.array([(int(x1[i]),int(y1[i])),(int(x2[i]),int(y2[i])),(int(x4[i]),int(y4[i])),(int(x3[i]),int(y3[i]))])
  #joining the four vertices on the image in their order ,True means closed figure
  #random integers are used as different colour codes,last attribute indicates thickness of rectangle boundary
  img=cv2.polylines(img,[pts],True,(r.randint(1,255),r.randint(5,255),r.randint(4,255)),3)
cv2.imshow("img",img) # show the image window with "img" as its title
cv2.waitKey() # infinite time delay  for waitkey
cv2.destroyAllWindows()  # destroy all windows

