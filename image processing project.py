#CAMSHIFT OBJECT TRACKING METHOD IN OPENCV
"""
import cv2
import numpy as np
cap=cv2.VideoCapture("highway.mp4")
#Take 1st frame
ret,frame=cap.read()
# setup initial location of window
x,y,w,h=250,150,100,50
track_window=(x,y,w,h)
# setup ROI for tracking
roi = frame[y:y+h,x:x+w]
hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
mask=cv2.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
roi_hist=cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# setup termination criteria either 10 iteration or move atleast 1 pt
term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT ,10,1)
cv2.imshow("roi",roi)
while(True):
    ret,frame=cap.read()
    if ret==True:
     hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     dst=cv2.calcBackProject([hsv_roi],[0],roi_hist,[0,180],1)
     #Apply meanshift to get the new location
     ret,track_window=cv2.CamShift(dst,track_window,term_criteria)
     #print(ret)
     #Draw it on image
     pts=cv2.boxPoints(ret)
     print(pts)
     pts=np.int0(pts)
     final_image=cv2.polylines(frame,[pts],True,(0,255,0),2)
     #x,y,w,h=track_window
     #final_image=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
     cv2.imshow("dst", dst)
     cv2.imshow("frame", final_image)
     k=cv2.waitKey(40) & 0xFF
     if k==27:
        break
    else:
     break
cap.release()
cv2.destroyAllWindows()
"""

#MEANSHIFT OBJECT TRACKING IN OPENCV
"""
import cv2
import numpy as np
cap=cv2.VideoCapture("highway.mp4")

#Take 1st frame
ret,frame=cap.read()
# setup initial location of window
x,y,w,h=400,200,100,50
track_window=(x,y,w,h)
# setup ROI for tracking
roi = frame[y:y+h,x:x+w]
hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
mask=cv2.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
roi_hist=cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# setup termination criteria either 10 iteration or move atleast 1 pt
term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT ,10,1)
cv2.imshow("roi",roi)
while(True):
    ret,frame=cap.read()
    if ret==True:
     hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     dst=cv2.calcBackProject([hsv_roi],[0],roi_hist,[0,180],1)
     #Apply meanshift to get the new location
     ret,track_window=cv2.meanShift(dst,track_window,term_criteria)
     #Draw it on image
     x,y,w,h=track_window
     final_image=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
     cv2.imshow("dst", dst)
     cv2.imshow("frame", final_image)

     k=cv2.waitKey(40) & 0xFF
     if k==27:
        break
    else:
     break
cap.release()
cv2.destroyAllWindows()
"""



"""
#BACKGROUND SUBTRACTION METHODS IN OPENCV 8:47:22
import cv2
import numpy as np
cap=cv2.VideoCapture("vtest.avi")
#fgbg=cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)
#kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg=cv2.createBackgroundSubtractorKNN()

while(True):
    ret,frame=cap.read()
    if frame is None:
     break
    fgmask=fgbg.apply(frame)
    #fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
    cv2.imshow("fgmask frame",fgmask)
    cv2.imshow("frame",frame)
    k=cv2.waitKey(30)
    if k=="q" or k==27:
        break
cap.release()
cv2.destroyAllWindows()

"""



#SHI TOMASI CORNER DETECTION METHOD
"""
import cv2
import numpy as np
img=cv2.imread("3500cropped.jpg")
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners=cv2.goodFeaturesToTrack(grey,100,0.00001,1)
corners=np.int0(corners)
for corner in corners:
    x,y=corner.ravel()
    cv2.circle(img,(x,y),4,(0,255,0),-1)
cv2.imshow("final",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""



#HARRIS CORNER DETECTOR
"""
import cv2
import numpy as np
img=cv2.imread("chessboard.png")
cv2.imshow("img",img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=np.float32(gray)#corner harris requires float 32 format
dst=cv2.cornerHarris(gray,2,5,0.04)
dst=cv2.dilate(dst,None)
print(dst.max())
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow("dst",img)
if cv2.waitKey(0) & 0xFF==27:
    cv2.destroyAllWindows()
"""






#FACE DETECTION IN VIDEO
"""
import cv2
from math import *
face_classifier=cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
eye_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)
while(True):
    cv2.namedWindow("FACE+EYES DETECTOR",cv2.WINDOW_NORMAL)
    ret,img=cap.read()

    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(grey,1.1,4)
    eyes=eye_classifier.detectMultiScale(grey,1.1,8)
    for face in faces:
        [x,y,w,h]=face
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        #cv2.circle(img,(int(x+0.5*w),int(y+0.5*h)),int((sqrt((w**2)+(h**2)))*0.5),(0,0,255),5)
    for eye in eyes:
        [x, y, w, h] = eye
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("FACE DETECTOR",img)
    k=cv2.waitKey(1) & 0xFF
    if k==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
"""

#FACE DETECTION  in IMAGE USING HAAR CASCADE CLASSIFIER
"""
import cv2
face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img=cv2.imread("latestphoto.jpg")
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_classifier.detectMultiScale(grey,1.1,4)
for face in faces:
    [x,y,w,h]=face
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("img",img)
cv2.waitKey()
"""





#CIRCLE DETECTION USING HOUGH LINES TRANSFORM
"""
import numpy as np
import cv2
img=cv2.imread("smarties.png")
output=img.copy()
grey=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
grey=cv2.medianBlur(grey,5)
circles=cv2.HoughCircles(grey,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
detected_circles=np.uint16(np.around(circles))
for (x,y,r) in detected_circles[0,:]:
    cv2.circle(output, (x, y), r, (0, 255, 0), 2)
    cv2.circle(output, (x, y), 2, (255, 255, 0), 2)

cv2.imshow("output",output)
cv2.waitKey()
cv2.destroyAllWindows()
"""

#LANE DETECTION PROJECT IN VIDEO
"""
import cv2
import numpy as np
cap=cv2.VideoCapture(0)
while(True):
   ret,img=cap.read()
   height=img.shape[0]
   width=img.shape[1]
   grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   canny = cv2.Canny(grey, 100, 120)
   black = np.zeros_like(canny)
   roi_vertices = np.array([(0, height), (int(width * 0.5), int(height * 0.5)), (width, height)])
   cv2.fillPoly(black, [roi_vertices], (255, 255, 255))
   crop = cv2.subtract(canny, cv2.bitwise_not(black))
   lines = cv2.HoughLinesP(crop, 2, (np.pi) / 60, 50, lines=np.array([]), minLineLength=25, maxLineGap=100)
   for line in lines:
       x1, y1, x2, y2 = line[0]
       cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
   cv2.imshow("houghlines", img)
   k=cv2.waitKey(1)
   if k==27:
    break
cap.release()
cv2.destroyAllWindows()
"""

#LANE DETECTION PROJECT IN IMAGE

"""
import cv2
import numpy as np
img=cv2.imread("road.png")
height=img.shape[0]
width=img.shape[1]
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
canny=cv2.Canny(grey,20,50)
black=np.zeros_like(canny)
roi_vertices=np.array([(0,height),(int(width*0.5),int(height*0.5)),(width,height)])
cv2.fillPoly(black,[roi_vertices],(255,255,255))
crop=cv2.subtract(canny,cv2.bitwise_not(black))
cv2.imshow("crop",crop)
lines=cv2.HoughLinesP(crop,6,(np.pi)/60,160,lines=np.array([]),minLineLength=10,maxLineGap=200)
for line in lines:
    x1,y1,x2,y2=line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),5)
cv2.imshow("houghlines",img)
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
"""
"""
cv2.polylines(canny,[roi_vertices],True,(255,255,255),2)

plt.imshow(crop)
"""
















"""
#THE PROBABILISTIC HOUGH TRANSFORM METHOD
import cv2
import numpy as np
img=cv2.imread("3500cropped.jpg")
imgrey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(imgrey,50,150,apertureSize=3)
cv2.imshow("edges",edges)
lines=cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=20)
print(lines[0])
for line in lines:
    x1,y1,x2,y2=line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow("Houghlines",img)
cv2.waitKey()
cv2.destroyAllWindows()
"""


"""
#THE  HOUGH TRANSFORM TECHNIQUE
import cv2
import numpy as np
img=cv2.imread("sudoku.png")
imgrey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(imgrey,50,150,apertureSize=3)
cv2.imshow("canny",edges)
lines=cv2.HoughLines(edges,1,np.pi/180,200)
for line in lines:
    rho,theta=line[0]
    a=np.cos(theta)
    b=np.sin(theta)
    x0=rho*a
    y0=rho*b
    x1=int(x0-1000*b)
    y1=int(y0+1000*a)
    x2=int(x0+1000*(b))
    y2=int(y0-1000*a)
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#TEMPLATE MATCHING IN OPENCV
"""
import cv2
import numpy as np
img=cv2.imread("messi5.jpg")
grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
template=cv2.imread("messi face.png",0)
w,h=template.shape[::-1]
res=cv2.matchTemplate(grey_img,template,cv2.TM_CCOEFF_NORMED)
#print((res))
threshold=0.53134
loc=np.where(res>=threshold)
print(loc)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img,pt,(pt[0]+w,pt[0]+h),(0,255,0),2)
cv2.imshow("img",img)
cv2.imshow("res",res)
cv2.waitKey()
cv2.destroyAllWindows()
"""

#HISTOGRAMS IN OPENCV

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
img=cv2.imread("lena.jpg",0)
#img = np.zeros((200,200),np.uint8)
#img=cv2.rectangle(img,(0,100),(200,200),(255,255,255),-1)
#img=cv2.rectangle(img,(0,50),(100,100),(127),-1)
#b,g,r=cv2.split(img)
#cv2.imshow("img",img)
#cv2.imshow("b",b)
#cv2.imshow("g",g)
#cv2.imshow("r",r)
hist=cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()

#plt.hist(b.ravel(),256,[0,256],label="b")
#plt.hist(g.ravel(),256,[0,256],label="g")
#plt.hist(r.ravel(),256,[0,256],label="r")
#plt.show()
"""

"""
#SHAPE DETECTION IN OPENCV
import cv2
import numpy as np
img=cv2.imread("shapes.jpg")
imgrey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,thresh=cv2.threshold(imgrey,240,255,cv2.THRESH_BINARY)
contours,_=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
for contour in contours:
    approx=cv2.approxPolyDP(contour,0.01* cv2.arcLength(contour,True),True)
    cv2.drawContours(img,[approx],0,(0,255,130),5)
    x=approx.ravel()[0]
    y=approx.ravel()[1]
    if len(approx) == 3:
        cv2.putText(img, "triangle", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 200, 0), 2)
    elif len(approx) == 4:
        x,y,w,h=cv2.boundingRect(approx)
        aspectratio=float(w)/h
        print(aspectratio)
        if aspectratio>=0.95 and aspectratio<=1.05:
            cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 200, 0), 2)
        else:
            cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 200, 0), 2)
    elif len(approx) == 5:
        cv2.putText(img, "pentagon", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 200, 0), 2)
    elif len(approx) == 6:
        cv2.putText(img, "hexagon", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 200, 0), 2)
    else:
        cv2.putText(img, "Dont know", (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 200, 0), 2)

cv2.imshow("img",img)
cv2.waitKey()
cv2.destroyAllWindows()
"""


"""
#BASIC MOTION DETECTION IN OPENCV
import cv2
cap=cv2.VideoCapture("vtest.avi")
ret,frame1=cap.read()
ret,frame2=cap.read()
while(True):
    diff=cv2.absdiff(frame1,frame2)
    gray=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    _,thresh=cv2.threshold(blur,30,255,cv2.THRESH_BINARY)
    dilated=cv2.dilate(thresh,None,iterations=3)
    contours,heirarchy=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    for contour in contours:
        (x,y,w,h)=cv2.boundingRect(contour)
        if cv2.contourArea(contour)<1000:
         continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,255,0),2,cv2.LINE_4)
        cv2.putText(frame1,"Status:{}".format("Movement"),(25,25),cv2.FONT_HERSHEY_DUPLEX,1,(0,130,255),3)
    #cv2.drawContours(frame1,contours,-1,(0,255,255),3)
    cv2.imshow("FRAME1",frame1)
    frame1=frame2
    ret,frame2=cap.read()
    k=cv2.waitKey(30)
    if k==27:
        break
cv2.destroyAllWindows()
cap.release()
"""
"""
#FIND AND DRAW CONTOURS
import cv2
import numpy as np
img=cv2.imread("3500.jpg")
img=cv2.pyrDown(img)
imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(imgray,100,255,cv2.THRESH_BINARY)
contours, heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#heirarchy is an optional vector containing information about topology of image
print("no. of contours="+ str(len(contours)))
print(contours[0])
cv2.drawContours(img,contours,-1,(0,255,0),2)

cv2.imshow("img",img)
cv2.imshow("imgray",imgray)

cv2.waitKey()
cv2.destroyAllWindows()
"""




"""
#IMAGE BLENDING USING PYRAMIDS
#Code not working
import cv2
import numpy as np
apple=cv2.imread("apple.jpg")
orange=cv2.imread("orange.jpg")
print(apple.shape)
print(orange.shape)
a_o=np.hstack((apple[:,:256],orange[:,256:]))
a_o2=np.vstack((apple[:256,:],orange[256:,:]))
#Gaussian pyramid for apple
apple_copy = apple.copy()
gp_apple = [apple_copy]
for i in range(6):
    apple_copy = cv2.pyrDown(apple_copy)
    gp_apple.append(apple_copy)
orange_copy = orange.copy()
gp_orange = [orange_copy]
#Gaussian pyramid for orange

for i in range(6):
    orange_copy = cv2.pyrDown(orange_copy)
    gp_orange.append(orange_copy)

#Laplacian pyramid for apple
apple_copy=gp_apple[5]
lp_apple=[apple_copy]
for i in range(5,0,-1):
    gauss_extended=cv2.pyrUp(gp_apple[i])
    laplacian=cv2.subtract(gp_apple[i-1],gauss_extended)
    lp_apple.append(laplacian)
#Laplacian pyramid for orange
orange_copy=gp_orange[5]
lp_orange=[orange_copy]
for i in range(5,0,-1):
    gauss_extended=cv2.pyrUp(gp_orange[i])
    laplacian=cv2.subtract(gp_orange[i-1],gauss_extended)
    lp_orange.append(orange)
#Now add left and right halves of image in each level
apple_orange_pyramid=[]
n=0
for apple_lap,orange_lap in zip(lp_apple, lp_orange):
    n=n+1
    cols,rows,ch=apple_lap.shape
    laplacian=np.hstack((apple_lap[:,:int(cols*0.5)], orange_lap[:,int(cols*0.5):]))
    apple_orange_pyramid.append(laplacian)
#reconstruct
apple_orange_reconstruct=apple_orange_pyramid[0]
for i in range(1,6):
    apple_orange_reconstruct=cv2.pyrUp(apple_orange_reconstruct)
    apple_orange_reconstruct=cv2.add(apple_orange_pyramid[i],apple_orange_reconstruct[i])

cv2.imshow("apple",apple)
cv2.imshow("orange",orange)
cv2.imshow("a_o",a_o)

cv2.imshow("a_o2",a_o2)
cv2.imshow("apple_orange_reconstruct",apple_orange_reconstruct)
cv2.waitKey()
cv2.destroyAllWindows()
"""







"""
#IMAGE PYRAMIDS
import cv2
img=cv2.imread("lena.jpg")
layer=img.copy()
gpa=[layer]
for i in range(6):
    layer=cv2.pyrDown(layer)
    gpa.append(layer)
    #cv2.imshow(str(i),layer)
layer=gpa[5]
cv2.imshow("upper level gaussian pyramid",layer)
lap=[layer]
for i in range(5,0,-1):
    gauss_extend=cv2.pyrUp(gpa[i])
    laplacian=cv2.subtract(gpa[i-1],gauss_extend)
    cv2.imshow(str(i),laplacian)
cv2.imshow("original",img)
"""
"""
lowr1=cv2.pyrDown(img)
lowr2=cv2.pyrDown(lowr1)
highr1=cv2.pyrUp(lowr2)
highr2=cv2.pyrUp(highr1)

cv2.imshow("img",img)
cv2.imshow("pyrdown1 img",lowr1)
cv2.imshow("pyrdown2 img",lowr2)
cv2.imshow("pyrup1 img",highr1)
cv2.imshow("pyrup2 img",highr2)
"""
"""
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
#CANNY EDGE DETECTION IN OPENCV
#ASSIGNMENT OF TRACKBARS COMPLETED IN cv2 WINDOW
import cv2
import numpy as np
from matplotlib import pyplot as plt
def f(x):
    pass
cv2.namedWindow("img")
cv2.createTrackbar("th1","img",0,255,f)
cv2.createTrackbar("th2","img",255,255,f)

while(True):

    img = cv2.imread("latestphoto.jpg", 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    th1=cv2.getTrackbarPos("th1","img")
    th2=cv2.getTrackbarPos("th2","img")
    canny = cv2.Canny(img, th1, th2)
    cv2.imshow("img",canny)

    k=cv2.waitKey(1)
    if k & 0xFF== ord("q"):
        break

cv2.destroyAllWindows()
"""
"""
"IMAGE GRADIENTS "
import numpy as np
import cv2
from matplotlib import pyplot as plt
img=cv2.imread("messi5.jpg",0)
lap=cv2.Laplacian(img,cv2.CV_64F,ksize=3)
lap=np.uint8(np.absolute(lap))
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0)
sobelx=np.uint8(np.absolute(sobelx))
sobely=cv2.Sobel(img,cv2.CV_64F,0,1)
sobely=np.uint8(np.absolute(sobely))
sobel=cv2.bitwise_or(sobelx,sobely)  #combnation of x and y
canny=cv2.Canny(img,100,200)
titles=["image","Laplacian","sobelx","sobely","sobel","canny"]
images=[img,lap,sobelx,sobely,sobel,canny]
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i],"gray")
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
"""
"""
#Smoothing and blurring images

import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("latestphoto.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
kern=np.ones((5,5),np.float32)/25
dst=cv2.filter2D(img,-1,kern)
blur=cv2.blur(img,(5,5))
gblur=cv2.GaussianBlur(img,(5,5),0)
median=cv2.medianBlur(img,5) #K-size must be odd always
biltralfilt=cv2.bilateralFilter(img,9,75,75)#PRESERVE EDGES
titles=["image","2D convolution","blur","gblur","median","biltralfilt"]
images=[img,dst,blur,gblur,median,biltralfilt]
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i],"gray")
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
"""


#MORPHOLOGICAL TRANSFORMATIONS IN OPENCV
"""[What is a pixel?
 Ans: an ultra small square containing only one colour.pixels with diff. colours form pics]"""
"""import cv2
import numpy as np
from matplotlib import pyplot as plt


img=cv2.imread("smarties.png",cv2.IMREAD_GRAYSCALE)
_,mask=cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)
kern=np.ones((5,5   ),np.uint8)
dilate=cv2.dilate(mask,kern,iterations=2)
erosion=cv2.erode(mask,kern,iterations=1)
opening=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kern) #first erosion then dilation
closing=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kern)#first dilation then erosion
grad=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,kern)#first dilation then erosion
cross=cv2.morphologyEx(mask,cv2.MORPH_CROSS,kern)#first dilation then erosion
titles=["image","mask","dilate","erosion","opening","closing","grad","cross"]
images=[img,mask,dilate,erosion,opening,closing,grad,cross]
for i in range(8):
    plt.subplot(2,4 ,i+1),plt.imshow(images[i],"gray")
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


"""
"""
#MATPLOTLIB WITH OPENCV
from matplotlib import pyplot as plt
import cv2
img=cv2.imread("messi5.jpg")
cv2.imshow("img",img)
img2=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img2)
#plt.xticks([]),plt.yticks([])
plt.show()

waitKey(0)
destroyAllWindows()
"""





"""
#ADAPTIVE THRESHOLDING IN OPENCV
from numpy import *
from cv2 import *
img=imread("sudoku.png",0)
_,th1=threshold(img,127,255,THRESH_BINARY)
th2=adaptiveThreshold(img,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,11,2)

th3=adaptiveThreshold(img,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,11,2)
imshow("img",img)
imshow("th1",th1)
imshow("th2",th2)

imshow("th3",th3)
waitKey()
destroyAllWindows()
"""

"""
#SIMPLE THRESHOLDING IN OPENCV

from numpy import *
from matplotlib import pyplot as plt
import cv2
img=cv2.imread("gradient.png")
_,th1=cv2.threshold(img,50,255,cv2.THRESH_BINARY)
_,th2=cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
_,th3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
_,th4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
_,th5=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
titles=["Original","BINARY","BINARY_INV","TRUNC","TOZERO","TOZERO_INV"]
images=[img,th1,th2,th3,th4,th5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],"gray")
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

#cv2.imshow("img",img)
#cv2.imshow("th1",th1)
#cv2.imshow("th2",th2)
#cv2.imshow("th3",th3)
#cv2.imshow("th4",th4)
#cv2.imshow("th5",th5)
cv2.waitKey()
cv2.destroyAllWindows()
"""





#HSV object detection in video
from numpy import *
from cv2 import *

def f(x):
 pass
cap=VideoCapture(0)
namedWindow("track")
createTrackbar("LH","track",0,255,f)
createTrackbar("LS","track",0,255,f)
createTrackbar("LV","track",0,255,f)
createTrackbar("UH","track",255,255,f)
createTrackbar("US","track",255,255,f)
createTrackbar("UV","track",255,255,f)

while(True):
 ret,frame=cap.read()
 hsv=cvtColor(frame,COLOR_BGR2HSV)
 lh = getTrackbarPos("LH", "track")
 ls = getTrackbarPos("LS", "track")
 lv = getTrackbarPos("LV", "track")
 uh = getTrackbarPos("UH", "track")
 us = getTrackbarPos("US", "track")
 uv = getTrackbarPos("UV", "track")
 low=array([lh,ls,lv])
 high=array([uh,us,uv])
 mask=inRange(hsv,low,high)
 res=bitwise_and(frame,frame,None,mask=mask)
 imshow("frame",frame)
 imshow("mask",mask)
 imshow("res",res)
 k=waitKey(1)
 if k & 0xFF==ord("q"):
  break
cap.release()
destroyAllWindows()


"""
#HSV OBJECT DETECTION IN AN IMAGE
from cv2 import *
from numpy import *
def nothing(x):
    pass

namedWindow("Track")
createTrackbar("LH","Track",0,255,nothing)
createTrackbar("LS","Track",0,255,nothing)
createTrackbar("LV","Track",0,255,nothing)
createTrackbar("UH","Track",255,255,nothing)
createTrackbar("US","Track",255,255,nothing)
createTrackbar("UV","Track",255,255,nothing)

while(True):

 frame=imread("smarties.png",1)
 hsv=cvtColor(frame,COLOR_BGR2HSV)
 lh = getTrackbarPos("LH", "Track")
 ls = getTrackbarPos("LS", "Track")
 lv = getTrackbarPos("LV", "Track")
 uh = getTrackbarPos("UH", "Track")
 us = getTrackbarPos("US", "Track")
 uv = getTrackbarPos("UV", "Track")

 lb=array([lh,ls,lv])
 ub=array([uh,us,uv])
 mask=inRange(hsv,lb,ub)
 #res=bitwise_and(frame,frame,None,mask=mask)
 imshow("frame",frame)
 imshow("mask",mask)
 imshow("res",res)
 k=waitKey(1)
 if k==27:
    break
destroyAllWindows()
"""



"""
#PYTHON OPENCV TRACKBAR2

from cv2 import *
from numpy import *
def nothing(x):
    print(x)

namedWindow("image")
switch="color/gray "
createTrackbar("CP", "image" , 10 , 400, nothing)
createTrackbar(switch, "image" , 0 , 1, nothing)

while(1):
    img = imread("messi5.jpg")
    pos = getTrackbarPos("CP", "image")
    img=putText(img,str(pos),(100,100),FONT_HERSHEY_COMPLEX,5,(0,255,255),1,LINE_8)

    s = getTrackbarPos(switch, "image")
    k=waitKey(1) & 0xFF

    if k==27:
      break


    if s==0:
     pass
    elif s==1:
     img=cvtColor(img,COLOR_BGR2GRAY)
    imshow("image", img)

destroyAllWindows()
"""
"""#PYTHON OPENCV TRACKBAR1

from numpy import *
from cv2 import *
def nothing(x):
    print(x)
img=zeros((300,512,3),uint8)
namedWindow("image")
switch="switch"
createTrackbar("B", "image" , 0 , 255, nothing)
createTrackbar("G", "image" , 0 , 255, nothing)
createTrackbar("R", "image" , 0 , 255, nothing)

createTrackbar(switch, "image" , 0 , 1, nothing)


while(1):
    imshow("image", img)
    b = getTrackbarPos("B", "image")
    g = getTrackbarPos("G", "image")
    r = getTrackbarPos("R", "image")
    s = getTrackbarPos(switch, "image")
    k=waitKey(1) & 0xFF
    if k==27:
      break
    if s==0:
     img[:]=[0,0,0]
    elif s==1:
     img[:]=[b, g, r]
destroyAllWindows()
"""

"""
#BITWISE OPERATORS
from cv2 import *
from numpy import *
img1=zeros((360,600,3),uint8)
img1=rectangle(img1,(250,50),(350,100),(255,255,255),-1)

img2=imread("baw.png",1)
#bitxor=bitwise_xor(img2,img1)
bnot=bitwise_not(img2)
imshow("baw",img1)
imshow("new",img2)
imshow("bitnot",bnot)

waitKey()
destroyAllWindows()
"""

"""
#Some basic operations on images
from cv2 import *
from numpy import *

img1=imread("copy1.jpg")
img2=imread("copy2.jpg")



print(img1.dtype)  #datatype
print(img1.shape)  #rows,columns,channels
print(img1.size)     #no. of pixels accessed

#ball=img1[280:340,330:390]
#img1[273:333,100:160]=ball


new=addWeighted(img1,0.9,img2,0.1,0)


imshow("add",new)

waitKey()
destroyAllWindows()
"""

"""
#mouse events handling
from cv2 import *
from numpy import *

#events=[i for i in dir(cv2) if "EVENT" in i]
#print(events)

def click(event,x,y,flags,param):
    if event==EVENT_LBUTTONDOWN:
        blue=img[y,x,0]
        green=img[y,x,1]
        red=img[y,x,2]
        circle(img,(x,y),3,(0,255,255),-1)
        colimg=zeros((500,500,3),uint8)
        colimg[:]=[blue,green,red]
        imshow("colimg",colimg)
img=imread("latestphoto.jpg",1)
imshow("img",img)

setMouseCallback("img",click)
"""
"""
def click(event,x,y,flags,param):
    if event==EVENT_LBUTTONDOWN:
        circle(img,(x,y),3,(0,255,255),-1)
        points.append((x,y))
        if len(points)>=2:
         line(img,points[-1],points[-2],(255,0,0),2)
         imshow("img",img)
#img=zeros((400,500,3),uint8)
img=imread("latestphoto.jpg",1)
imshow("img",img)
points=[]

setMouseCallback("img",click)
"""


""" 
def click(event,x,y,flags,param):
    if event==EVENT_LBUTTONDOWN:
     print(x,",",y)
     putText(img,str(x)+","+str(y),(x,y),FONT_ITALIC,0.5,(200,0,0),2)
     imshow("img",img)
    if event==EVENT_RBUTTONDOWN:
     blue=img[y,x,0]
     green=img[y,x,1]
     red=img[y,x,2]
     print(blue,green,red)
     putText(img,str(blue)+","+str(green)+","+str(red),(x,y),FONT_ITALIC,0.5,(200,0,0),2)
     imshow("img",img)
img=imread("latestphoto.jpg",1)
#img=zeros((500,500,3),uint8)
imshow("img",img)                     
setMouseCallback("img",click)
waitKey()
destroyAllWindows()
"""

""" #Add text in videos
from cv2 import *
from datetime import *

cap=VideoCapture("video1.mp4")
#cap.set(CAP_PROP_FRAME_WIDTH,3000)
#cap.set(CAP_PROP_FRAME_HEIGHT,3000)

a=cap.get(CAP_PROP_FRAME_WIDTH)
b=cap.get(CAP_PROP_FRAME_HEIGHT)

while(True):
    ret,frame=cap.read()
    if ret==True:
     c="width="+str(a)+",height="+str(b)
     dt=str(datetime.now())
     frame=putText(frame,dt,(80,129),FONT_ITALIC,1,(255,0,0),2)   
     imshow("frame",frame)
     if waitKey(1) & 0xFF ==ord('q'):
      break
    else:
     break


cap.release()
destroyAllWindows()

"""




"""
#learn about shapes and objects

from numpy import *

from cv2 import *

img=zeros([512,512,3],uint8)
#img=imread("latestphoto.jpg",0)


img=line(img,(80,12),(220,12),(147,55,3),6)
img=arrowedLine(img,(80,200),(220,200),(0,55,3),6)   #color in BGR format
img=rectangle(img,(60,12),(220,200),(147,55,3),4)   #-1 in place of 4 will fill space
img=circle(img,(140,106),100,(147,55,3),4)
img=putText(img,"shivaye",(80,29),FONT_ITALIC,1,(0,255,0),4,LINE_8)
imshow("photo",img)

waitKey()
destroyAllWindows()

"""


"""
#learn about video projects
from cv2 import *


cap=VideoCapture("video1.mp4")
fourcc=VideoWriter_fourcc(*"MP4V")
out=VideoWriter("outputvid.mp4",fourcc,20,(1920,1080))
while(cap.isOpened()):
    ret,frame=cap.read()  #true or false saved in ret,frame will be saved in frame read by cap
    if ret==True:
     print(cap.get(CAP_PROP_FRAME_WIDTH))
     print(cap.get(CAP_PROP_FRAME_HEIGHT))
     
     new=cvtColor(frame,COLOR_BGR2HLS)
     imshow("frame",new)
     out.write(new)
     if waitKey(1) & 0xFF ==ord('q'):
      break
    else:
     break
cap.release()
out.release()
destroyAllWindows()
"""
   
"""
#read,write,save image 
from cv2 import *

img=imread("latestphoto.jpg",1)
print(img)
new=cvtColor(img,COLOR_BGR2LAB)
imshow("myphoto",new)
k=waitKey()
if k==9:
  destroyAllWindows()
elif ord("s"):
  imwrite("copycheck.jpg",new)
  destroyAllWindows() 
"""
