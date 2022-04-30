#CENTER TRACKING OF A RING

import cv2
import pandas as pd
import numpy as np
cap=cv2.VideoCapture("Dynamics.mov") #capture video in local directory
file=pd.read_excel(r"F:\summer project\image processing\trajectory_excel_imp\ringcrp9.xlsx")# read excel file for the 9th ring
x=file["x"] # column of the x coordinate of the centre of ring
y=file["y"] # column of the y coordinate of the centre of ring
i=0  # i is frame index ,initialised to zero
while(cap.isOpened()):  # while loop runs as long as video plays
    ret,frame=cap.read() # read frames command for video,here ret is True if there is any frame present at a particular time instant
    # make a red circle with radius scale 5 on the video frame with index i with x and y as center coordinates
    # -1 means the red color fills the complete inner part of circle
    cv2.circle(frame,(int(x[i]),int(y[i])),5,(0,0,255),-1)
    cv2.imshow("frame",frame) #show the video frame
    i=i+1 #increment of the video frame index
    k=cv2.waitKey(4) # time delay of 4 milliseconds
    if k & 0xFF==ord("q"): #pressing keyboard key "q" will terminate the whole operation
        break
cap.release() # release all the resources of the video
cv2.destroyAllWindows() # release all the resources of the video


"""
#FRAME EXTRACTION FROM VIDEO
import cv2
cap=cv2.VideoCapture("Dynamics.mov")  #capture video in local directory
i=0  # i is frame index ,initialised to zero
while(cap.isOpened()): # while loop runs as long as video plays
    ret,frame=cap.read()   # read frames command for video,here ret is True if there is any frame present at a particular time instant
    cv2.imshow("frame", frame) # show video frame
    print(i) #print frame index
    cv2.imwrite("frame"+str(i)+".jpg", frame) # write the frame as jpg file in local directory,index is also mentioned in naming it
    i=i+1 #increment of the video frame index
    k=cv2.waitKey(1) # time delay of 1 millisecond
    if k & 0xFF==ord("q"): #pressing keyboard key "q" will terminate the whole operation
        break
cap.release() # release all the resources of the video
cv2.destroyAllWindows() # destroy all windows after completion of video resources release
"""