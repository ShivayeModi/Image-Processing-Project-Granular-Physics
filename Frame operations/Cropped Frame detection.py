#SINGLE RING TRACKING IN VIDEO WITH CROPPED RECTANGLE
import cv2
import numpy as np
import pandas as pd
cap=cv2.VideoCapture("Dynamics.mov")  #capture video in local directory
file=pd.read_excel(r"F:\summer project\image processing\trajectory_excel_imp\ringcrp9.xlsx") #read excel file
x0=file["x"]# x coordinate of center of ring
y0=file["y"]#y coordinate of center of ring
a=file["a"].mean()# mean of major axis column
b=file["b"].mean()# mean of minor axis column
phi=((np.pi)*(-1)*file["phi"])/180 # phi is taken in radians,negative value is due to cv window y-axis inversion
phid=(-1)*file["phi"] #phi in degrees
# calculation of the vertices of the rectangle in conventional cartesian coordinate system
x1=x0+(0.5*a*(np.sin(phi)))-(0.5*b*(np.cos(phi)))
y1=y0+(0.5*a*(np.cos(phi)))+(0.5*b*(np.sin(phi)))
x2=x0+(0.5*a*(np.sin(phi)))+(0.5*b*(np.cos(phi)))
y2=y0+(0.5*a*(np.cos(phi)))-(0.5*b*(np.sin(phi)))
x4=x0-(0.5*a*(np.sin(phi)))+(0.5*b*(np.cos(phi)))
y4=y0-(0.5*a*(np.cos(phi)))-(0.5*b*(np.sin(phi)))
x3=x0-(0.5*a*(np.sin(phi)))-(0.5*b*(np.cos(phi)))
y3=y0-(0.5*a*(np.cos(phi)))+(0.5*b*(np.sin(phi)))
i=0 # i is frame index ,initialised to zero
while(True):    # infinte loop starts
    ret,frame=cap.read()  # read frames command for video,here ret is True if there is any frame present at a particular time instant
    black=np.zeros_like(frame) # create black image of the same size as that of video frame
    # declaration of array of four vertices
    pts = np.array([(int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (int(x4[i]), int(y4[i])), (int(x3[i]), int(y3[i]))])
    # joining the four vertices on the video frame in their order ,True means closed figure,white color boundary,thickness scale 2
    cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
    ## fill the closed polygon with white color on the black image
    cv2.fillPoly(black,[pts],(255,255,255))
    # to get cropped image perform masking operation by adding original image and bitwise not of black image
    crop=cv2.add(frame,cv2.bitwise_not(black))
    ################################
    rotmatrix=cv2.getRotationMatrix2D((int(x0[i]),int(y0[i])),-1*(90+int(phid[i])),1)
    rotrect=cv2.warpAffine(crop,rotmatrix,(frame.shape[1],frame.shape[0]))
    roi=rotrect[int(y0[i])-int(0.5*b):int(y0[i])+int(0.5*b),int(x0[i])-int(0.5*a):int(x0[i])+int(0.5*a)]
    cv2.imshow("frame",frame)
    cv2.imshow("black", black)
    cv2.imshow("crop",crop)
    cv2.imshow("rotrect",rotrect)
    cv2.imshow("roi",roi)
    cv2.imwrite("ring9_roi_frame_"+str(i)+".jpg",roi)
    k=cv2.waitKey(4) # time delay of 4 milliseconds
    i=i+1 #increment of the video frame index
    if k & 0xFF==ord("q"): #pressing keyboard key "q" will terminate the whole operation
        break
cap.release() # release all the resources of the video
cv2.destroyAllWindows() # destroy all windows after completion of video resources release

