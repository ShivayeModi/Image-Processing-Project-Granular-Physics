
#Matching of the center of rings
from pandas import *                # import all functions from pandas library
from matplotlib import pyplot as plt
import cv2
img=cv2.imread("3500.jpg")        #read the image
file=read_excel(r"F:\summer project\image processing\coordinates data.xlsx")    #specify the file path where data of rings coordinates is given
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # change color format from BGR to RGB as matplotlib works in RGB mode
plt.imshow(img)      #show the image in matplotlib window
plt.scatter(file["X"],file["Y"],color="r") #the X column of file contains x coordinates,Y column contains y coordinates,plotted in scatter format
plt.show()  #show the matplotlib window
