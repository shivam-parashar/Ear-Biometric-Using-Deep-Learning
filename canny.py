import cv2
import numpy as np

gray = cv2.imread('ex2.bmp')
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

#canny
img_canny = cv2.Canny(gray,60,120)
point1=0
point2=0
point3=0
point4=0
x1=0
x2=0
x3=0
x4=0
y1=0
y2=0
y3=0
y4=0


for i in range(0,180):
	for j in range(0,50):
		if img_canny[i][j]==255:
			if i<90:
				if j>24:
					x1 += i;
					y1 += j;
					point1 += 1;
				else : 
					x2 += i;
					y2 += j;
					point2 += 1;
			else :
				if j>24:
					x3 += i;
					y3 += j;
					point3 += 1;
				else : 
					x4 += i;
					y4 += j;
					point4 += 1; 
			
			
			
						
			
centroidx1 = x1/point1;
centroidy1 = y1/point1;

centroidx2 = x2/point2;
centroidy2 = y2/point2;

centroidx3 = x3/point3;
centroidy3 = y3/point3;

centroidx4 = x4/point4;
centroidy4 = y4/point4;




print(centroidx1)
print(centroidy1)
print(point1)


cv2.imshow("Original Image", gray)
cv2.imshow("Canny", img_canny)





cv2.waitKey(0)
cv2.destroyAllWindows()
