#cropping of all rectangles from a single image
import cv2
from pandas import *
import numpy as np
file=read_excel(r"F:\summer project\image processing\coordinates data.xlsx") # read excel file
x0=file["X"] # x coordinate of center of ring
y0=file["Y"] #y coordinate of center of ring
a=file["a"].mean()  # mean of major axis column
b=file["b"].mean()  # mean of minor axis column
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
 # three different empty cv windows fit to computer screen
 cv2.namedWindow("rect" + str(i), cv2.WINDOW_NORMAL)
 cv2.namedWindow("white"+str(i), cv2.WINDOW_NORMAL)
 cv2.namedWindow("crop"+str(i),cv2.WINDOW_NORMAL)
 black=np.zeros([1080,1920,3],np.uint8) #create a black image of the 1080 pixel height ,1920 pixel length
 img = cv2.imread("3500.jpg")
 # declaration of array of four vertices
 pts=np.array([(int(x1[i]),int(y1[i])),(int(x2[i]),int(y2[i])),(int(x4[i]),int(y4[i])),(int(x3[i]),int(y3[i]))])
 # joining the four vertices on the image in their order ,True means closed figure,white color boundary
 img=cv2.polylines(img,[pts],True,(255,255,255),3)
 white=cv2.fillPoly(black,[pts],(255,255,255))  # fill the closed polygon with white color
 # to get cropped image perform masking operation by adding original image and bitwise not of white patch
 crop=cv2.add(img,cv2.bitwise_not(white))
 cv2.imshow("white"+str(i),white)  #show white patch
 cv2.imshow("rect" + str(i), img)  #show  rectangle with white boundary on original image
 cv2.imshow("crop" + str(i), crop) #show cropped image
 cv2.imwrite("rect"+str(i)+".jpg",crop)  #write the cropped image as jpg file
 cv2.waitKey(9000) # time delay 9000 milliseconds means 9 seconds
 cv2.destroyAllWindows() # destroy all windows

