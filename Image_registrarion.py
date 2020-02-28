import cv2
import numpy as np
import os
import glob
good_percent=0.15
orgipath="Original/original.jpg"

imgor = cv2.imread(orgipath)
folder="Test_images/*.jpg"
for filename in glob.glob("Test_images\*.jpg"):
 img = cv2.imread(filename)

 imgorGray=cv2.cvtColor(imgor,cv2.COLOR_BGR2GRAY)
 imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

 ORB = cv2.ORB_create(500)
 Point_src,desc1 = ORB.detectAndCompute(imgGray,None)
 Point_dest,desc2 = ORB.detectAndCompute(imgorGray,None)

 matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
 match = matcher.match(desc1,desc2,None)

 match.sort(key=lambda x: x.distance, reverse=False)

 noOfgood = int((len(match)*good_percent))

 match=match[:noOfgood]

 Point_src1 = np.zeros((len(match),2),dtype=np.float32)
 Point_dest1 = np.zeros((len(match),2),dtype=np.float32)

 for i,j in enumerate(match):
    Point_src1[i, :] = Point_src[j.queryIdx].pt
    Point_dest1[i, :] = Point_dest[j.trainIdx].pt

 h = cv2.findHomography(Point_src1,Point_dest1,cv2.RANSAC)[0]

 height,width,rgb=imgor.shape

 im1Reg = cv2.warpPerspective(img, h, (width, height))
 filename=filename[12:]
 name="output/"+filename
 print(name)
 cv2.imwrite(name,im1Reg)
