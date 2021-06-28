import numpy as np
import cv2

input_ =  cv2.imread(r'C:\Users\skgha\Projects\Leaf_Disease_Detection\a.JPG')


def bg_remove(image):
        img = cv2.GaussianBlur(image,(5,5),0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # find the green color 
        mask_green = cv2.inRange(hsv, (36,0,0), (86,255,255))
        
        # find the brown color
        mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
        
        # find the yellow color in the leaf
        mask_yellow = cv2.inRange(hsv, (14, 39, 64), (40, 255, 255))
        
        # find any of the three colors(green or brown or yellow) in the image
        mask = cv2.bitwise_or(mask_green, mask_brown)
        mask = cv2.bitwise_or(mask, mask_yellow)
        
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img,img, mask= mask)
        img[np.where((res==[0,0,0]).all(axis=2))] = [255,255,255]

        cv2.imshow("Final_Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
