#Sample image creation
"""
import numpy as np
import cv2
img=np.zeros([100,100])
img=cv2.rectangle(img,(60,60),(80,80),(255,255,255),-1)
cv2.imwrite("rect2.jpg",img)
cv2.imshow("img",img)
cv2.waitKey()
"""

 # Match the template
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random as r
img=cv2.imread("3000.jpg")
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
temp=cv2.imread("3010.jpg",0)
h,w=temp.shape[:]
print(h,w)
res=cv2.matchTemplate(grey,temp,cv2.TM_CCORR_NORMED)
print(res.shape)
#thresh=0
#loc=np.where(res>=thresh) #returns the indices of elements in an input array where the given condition is satisfied.
#print(loc)
#cv2.imshow("img",img)
#cv2.waitKey(5000)
#for pt in zip(*loc[::-1]): #unzipping the loc coordinates in reverse manner
#    cv2.rectangle(img,pt,(pt[0]+w,pt[1]+h), (r.randint(1,255),r.randint(1,255) ,r.randint(1,255)), 1)

min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
top_left=max_loc
bottom_right=(top_left[0]+w,top_left[1]+h)
cv2.imshow("matched_win",img)
plt.imshow(res,cmap="gray")
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
"""



 # Detect corners using Shi Tomasi corner detector
"""
import cv2
img=cv2.imread("ring_20frame_28.jpg")
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners=cv2.goodFeaturesToTrack(grey,1,0.001,10)
for corner in corners:
    x,y=corner.ravel()
    cv2.imshow("img",img)
    cv2.waitKey(3000)
    cv2.circle(img,(int(x),int(y)),1,(0,255,0),2)
cv2.imshow("detect",img)
cv2.waitKey()
"""
