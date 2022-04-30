"""
import pandas as pd
import matplotlib.pyplot as plt
file=pd.read_csv("F:\\summer project\\image processing\\Trajectories_only_crop_new"+"\\"+"newringcrp"+str(40)+".txt")
frame=file["frame_no"]
x=file["x"]
plt.plot(frame,x)
plt.show()
"""
#CURVE FIT AND AVERAGING METHOD
import cv2

"""
from matplotlib import pyplot as plt
import numpy as np
x=[i for i in range(1,22)]
y=[1,2,3,4,5,6,7,8,9,10,11,11,9,8,7,6,5,4,3,2,1]
xnew=np.linspace(1,22,220)

maxima=max(y)
max_loc=y.index(maxima)
max_points=[]
for i in range(max_loc-5,max_loc+5):
    if y[i]==maxima:
        max_points.append(i)
print(max_points)
print(np.mean(max_points))


curve=np.polyfit(x,y,5) #return coeff of polynomial
poly=np.poly1d(curve)
print(poly(x))
print(max(poly(xnew)))
plt.scatter(x,y)
plt.plot(xnew,poly(xnew),color="r")
plt.plot()
plt.show()


"""

"""
import matplotlib.pyplot as plt

a=[1.05671771e+04,7.90383583e+03,1.15053737e+04,2.69238158e+04,4.01654871e+04,3.00164875e+04,8.47225375e+03,1.84213250e+03,
   5.56397792e+03 ,6.76030667e+03, 2.54727708e+03,-1.21161500e+03
]
b=[i for i in range(len(a))]
plt.plot(b,a)
plt.show()

import cv2
import numpy as np
import pandas as pd
"""



import pandas as pd
img=cv2.imread("F:\\summer project\\image processing\\Frame operations"+"\\"+"frame0"+".jpg")
cv2.namedWindow("img",cv2.WINDOW_NORMAL)
for i in range(0,175): # i as ring index
    file = pd.read_csv("F:\\summer project\\image processing\\Trajectories_only_crop_new\\newringcrp"+str(i)+".txt")
    frames_count=file.shape[0] # no. of frames data in a typical ring txt file
    x0 = file["x"]
    y0 = file["y"]
    a = 80
    b = 18
    cv2.circle(img,(int(x0[0]),int(y0[0])),3,(0,0,255),-1)
    cv2.putText(img,str(i),(int(x0[0]),int(y0[0])),cv2.FONT_ITALIC,1,(0,255,0),2)
cv2.imshow("img",img)
cv2.imwrite("index_of_rings.jpg",img)
cv2.waitKey()


#matrix operations
"""
import cv2
import numpy as np
a=np.array([35,9,65,49,36,45,87,91,24,39,56,98])
a=a.reshape(3,4)
print(a)
maxloc=np.where(a==np.max(a))
y=int(maxloc[0]+1)
x=int(maxloc[1]+1)
print(y,x)
corr_offset=[y-a.shape[0],x-a.shape[1]]
print(corr_offset)
"""


#ROTATE THE IMAGE
"""
import cv2
import numpy as np
img=cv2.imread("messi5.jpg")
#imgr=cv2.rotate(img,rotateCode=2)
h,w,c=img.shape
matrix=cv2.getRotationMatrix2D((int(w/2),int(h/2)),45,1.0)
rotated=cv2.warpAffine(img,matrix,(w,h))
cv2.imshow("rotator",rotated)
cv2.waitKey(0)
"""



#HSV COLOR RANGE USAGE FAILED
"""
import cv2
import numpy as np
cv2.namedWindow("crop")
img=cv2.imread("desired area.jpg")
lblue=np.array([110,50,50])
hblue=np.array([130,255,255])
mask=cv2.inRange(img,lblue,hblue)
cv2.imshow("crop",mask)
cv2.waitKey()
"""

#Approach second continued
"""
import cv2
import numpy as np
img=cv2.imread("desired area.jpg")
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,th=cv2.threshold(grey,127,255,cv2.THRESH_BINARY)
morph=cv2.morphologyEx(th,cv2.MORPH_OPEN,())

cv2.imshow("img",img)
cv2.waitKey()
cv2.destroyAllWindows()
"""
""" #nested for loop approach
import cv2
import numpy as np
cv2.namedWindow("crop")
img=cv2.imread("3500cropped.jpg")
h=img.shape[0]
w=img.shape[1]

pts=[(50,50),(100,150),(200,50),(300,30)]
pts=np.array(pts)
img=cv2.polylines(img,[pts],True,(0,255,0),2)
img_pts=[]
for x in range(w):
 for y in range(h):
  dist = cv2.pointPolygonTest(pts, (x, y), False)
  if dist==1:
   img_pts.append(img[y,x])
  else:
   continue
pixel_pts=np.array(img_pts,dtype=np.uint8)
print(pixel_pts)


cv2.imshow("area",img)
cv2.waitKey()
cv2.destroyAllWindows()
"""

"""
import cv2
import numpy as np
img=cv2.imread("logo.png")
black=np.zeros_like(img)
white=cv2.bitwise_not(black)
sum=cv2.subtract(white, black)
cv2.imshow("sum",sum)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
import random
a=random.randint(1,255)
print(a)
"""

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

cv2.namedWindow("img",cv2.WINDOW_NORMAL)
img=cv2.imread("3500.jpg")
cnt=np.array([(874,1021),(875,1003),(782,995),(781,1013)])
cv2.polylines(img,[cnt],True,(0,255,0),5)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""