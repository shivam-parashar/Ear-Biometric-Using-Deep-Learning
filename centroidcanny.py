import cv2
import numpy as np

gray = cv2.imread('017_2.bmp')
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

#canny
img_canny = cv2.Canny(gray,60,120)
point1=0,point2=0,point3=0,point4=0
x=0
y=0

for i in range(0,180):
	for j in range(0,50):
		if img_canny[i][j]==255:
			x += i;
			y += j;
			point += 1;
			
			
						
			
centroidx = x/point;
centroidy = y/point;


print(centroidx)
print(centroidy)
print(point)


cv2.imshow("Original Image", gray)
cv2.imshow("Canny", img_canny)










cv2.waitKey(0)
cv2.destroyAllWindows()
