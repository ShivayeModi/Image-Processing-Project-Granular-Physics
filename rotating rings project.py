#ROTATING RINGS PROJECT

#Contour tracing of all rings

import cv2
import numpy as np
def f(x):
    pass
cv2.namedWindow("contour")
cv2.createTrackbar("var","contour",0,255,f)
while(True):
 img = cv2.imread("3500cropped.jpg")
 img = cv2.pyrDown(img)
 imgrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 var=cv2.getTrackbarPos("var","contour")
 _,th=cv2.threshold(imgrey,var,255,cv2.THRESH_BINARY)
 contour,hierarchy=cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
 print(len(contour))
 cv2.drawContours(img,contour,-1,(0,255,0),2)
 cv2.imshow("contour",img)
 k=cv2.waitKey(1) & 0xFF
 if k==27:
  break
cv2.destroyAllWindows()


"""
#ROTATED RECTANGLE creation
import cv2
import numpy as np
import random
cv2.namedWindow("rings")
img=cv2.imread("3500cropped.jpg",1)
img=cv2.pyrDown(img)
imgrey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,thresh=cv2.threshold(imgrey,53,255,cv2.THRESH_BINARY)
contour,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
print(contour[2])
rand_no=random.randint(0,len(contour))
rect=cv2.minAreaRect(contour[rand_no])
box=cv2.boxPoints(rect)
box=np.int0(box)
img=cv2.drawContours(img,[box],-1,(0,0,255),5)
cv2.imshow("rings",img)
cv2.waitKey()
cv2.destroyAllWindows()
"""
"""
#POLYGON CREATION AROUND REGION OF INTEREST

import cv2
import numpy as np
points=[]
def click(event,x,y,flags,param):
  if event==cv2.EVENT_LBUTTONDOWN:
      cv2.circle(img,(x,y),1,(0,0,255),5)
      points.append((x,y))
      cv2.imshow("img", img)
  if len(points)>=2:
   for i in range(1,len(points)):
    cv2.imshow("img", img)
    if len(points)==5:
     print(points)
     pts = [list(points[i]) for i in range(5)]
     print(pts)
     pts=np.array(pts,np.int32)
     print(pts)
     cv2.polylines(img,[pts],True,(0,255,0),2)
     cv2.fillPoly(white,[pts],(255,255,255))
     cv2.imshow("img",img)
     cv2.imshow("white",white)
     res=cv2.add(img,cv2.bitwise_not(white))
     cv2.imshow("res",res)
     cv2.imwrite("desired area.jpg",res)
img=cv2.imread("3500cropped.jpg")
img=cv2.pyrDown(img)
print(img.shape)
white=np.zeros([512,536,3],np.uint8)
cv2.imshow("img",img)
cv2.setMouseCallback("img",click)
cv2.waitKey()
cv2.destroyAllWindows()
"""
