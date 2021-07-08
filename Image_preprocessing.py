import cv2
import numpy as np

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
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #plt.imshow(img)
        
        return img

def main(img):

    img = cv2.imread(r'C:\Users\skgha\Projects\Leaf_Disease_Detection\a.JPG')
    img = cv2.resize(img, ((int)(img.shape[1]/5), (int)(img.shape[0]/5)))
    original = img.copy()
    neworiginal = img.copy()
    #plt.imshow(img)

    p  = 0 
    for i in range(img.shape[0]):
    	for j in range(img.shape[1]):
    		B = img[i][j][0]
    		G = img[i][j][1]
    		R = img[i][j][2]
    		if (B > 110 and G > 110 and R > 110):
    			p += 1
    totalpixels = img.shape[0]*img.shape[1]
    per_white = 100 * p/totalpixels
    #excluding all the pixels with colour close to white if they are more than 10% in the image
    if per_white > 10:
    	img[i][j] = [200,200,200]
    	#cv2.imshow('color change', img)
    
    
    #Guassian blur
    blur1 = cv2.GaussianBlur(img,(3,3),1)

    #mean-shift algo
    newimg = np.zeros((img.shape[0], img.shape[1],3),np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 10 ,1.0)
    
    img = cv2.pyrMeanShiftFiltering(blur1, 20, 30, newimg, 0, criteria)
    #cv2.imshow('means shift image',img)
    
    #Guassian blur
    blur = cv2.GaussianBlur(img,(11,11),1)
    
    #Canny-edge detection
    canny = cv2.Canny(blur, img.shape[1], img.shape[0])
    
    canny = cv2.cvtColor(canny,cv2.COLOR_GRAY2BGR)
    
    #contour to find leafs
    bordered = cv2.cvtColor(canny,cv2.COLOR_BGR2GRAY)
    contours,hierarchy = cv2.findContours(bordered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    global maxid
    maxC = 0
    for x in range(len(contours)):													#if take max or one less than max then will not work in
    	if len(contours[x]) > maxC:													# pictures with zoomed leaf images
    		maxC = len(contours[x])
    		maxid = x
            
    
    perimeter= cv2.arcLength(contours[maxid],True)
    #print perimeter
    Tarea = cv2.contourArea(contours[maxid])
    cv2.drawContours(neworiginal,contours[maxid],-1,(0,0,255))
    #cv2.imshow('Contour',neworiginal)
    
    #Creating rectangular roi around contour
    height, width, _ = canny.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    frame = canny.copy()
    
    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
    	(x,y,w,h) = cv2.boundingRect(contours[maxid])
    	min_x, max_x = min(x, min_x), max(x+w, max_x)
    	min_y, max_y = min(y, min_y), max(y+h, max_y)
    	if w > 80 and h > 80:
    		#cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)   #we do not draw the rectangle as it interferes with contour later on
    		roi = img[y:y+h , x:x+w]
    		originalroi = original[y:y+h , x:x+w]
    		
    if (max_x - min_x > 0 and max_y - min_y > 0):
    	roi = img[min_y:max_y , min_x:max_x]	
    	originalroi = original[min_y:max_y , min_x:max_x]
    	#cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)   #we do not draw the rectangle as it interferes with contour
    
    #cv2.imshow('ROI', frame)
    #cv2.imshow('rectangle ROI', roi)
    img = roi
    #Changing colour-space
    #imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imghls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
    #cv2.imshow('HLS', imghls)
    imghls[np.where((imghls==[30,200,2]).all(axis=2))] = [0,200,0]
    #cv2.imshow('new HLS', imghls)
    
    #Only hue channel
    huehls = imghls[:,:,0]
    #cv2.imshow('img_hue hls',huehls)
    #ret, huehls = cv2.threshold(huehls,2,255,cv2.THRESH_BINARY)
    
    huehls[np.where(huehls==[0])] = [35]
    #cv2.imshow('img_hue with my mask',huehls)
    
    #Thresholding on hue image
    ret, thresh = cv2.threshold(huehls,28,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow('thresh', thresh)
    
    #Masking thresholded image from original image
    mask = cv2.bitwise_and(originalroi,originalroi,mask = thresh)
    #cv2.imshow('masked out img',mask)
    
    #Finding contours for all infected regions
    contours,heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    Infarea = 0
    for x in range(len(contours)):
    	cv2.drawContours(originalroi,contours[x],-1,(255,0,0))
    	#plt.imshow(originalroi)
    	
    	#Calculating area of infected region
    	Infarea += cv2.contourArea(contours[x])
    
    if Infarea > Tarea:
    	Tarea = img.shape[0]*img.shape[1]
    
    print ('_______________________\n| Total area: ' + str(Tarea) + '   |\n|_____________________|')
    
    #Finding the percentage of infection in the leaf
    print ('\n__________________________\n| Infected area: ' + str(Infarea) + ' |\n|________________________|')
    
    try:
    	per = 100 * Infarea/Tarea
    
    except ZeroDivisionError:
    	per = 0
    
    print( '\n_________________________________________________\n| Percentage of infection region: ' + str(per) + ' % |\n|_______________________________________________|')
    

input_ =  cv2.imread(r'C:\Users\skgha\Projects\Leaf_Disease_Detection\a.JPG')
img = bg_remove(input_)
main(img)