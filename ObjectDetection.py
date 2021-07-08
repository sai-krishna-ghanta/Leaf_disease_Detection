import cv2 as cv
import numpy as np

#Path of the input image 
path = r"C:\Users\skgha\Projects\Leaf_Disease_Detection\ObjectDetection_Input.jpg"
# Copy the neural network architecture
cfg_file = r'C:\Users\skgha\Projects\OCR\yolo-leaf-detector_weights\yolo-leaf-detector_weights\yolov3-tiny-obj.cfg'
# Copy the pre-trained weights
weight_file = r'C:\Users\skgha\Projects\OCR\yolo-leaf-detector_weights\yolo-leaf-detector_weights\yolov3-tiny-obj_final.weights'
# Copy the names of the classes
namesfile = r'C:\Users\skgha\Projects\OCR\yolo-leaf-detector_weights\yolo-leaf-detector_weights\obj.names'

#loading yolov3 by passing weights and cfg files
net = cv.dnn.readNet(weight_file,cfg_file)
classes=[]
with open(namesfile,"r") as f:
    classes = [line.strip() for line in f.readlines()]

model = cv.dnn_DetectionModel(net)
net.getUnconnectedOutLayers()
layer_names = net.getLayerNames()
outputLayers= [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes),3))

img = cv.imread(path)
blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
swapRB=False, crop=True)
net.setInput(blob)
outs = net.forward(outputLayers)
#finding confidence score of algorithm in object detection in blob
class_ids=[]
confidences =[]
boxes =[]
h, w = img.shape[:2]
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        #object detected
        if confidence > 0.05:
            center_x= int(detection [0] * w)
            center_y= int(detection [1] * h)
            ww = int(detection [2] * w)
            hh = int(detection [3] * h)
            
            #coordinates of a rectangle
            x= int(center_x - ww/2)
            y= int(center_y - hh/2)
            
            boxes.append([x,y,ww,hh])
            confidences.append(float(confidence)) 
            class_ids.append(class_id)

#to eliminate multiple detections for same object
#using non - max suppression
#anything with IOU_threshold < 0.6 will be removed
indexes = cv.dnn.NMSBoxes(boxes,confidences,0.4,0.6) 


# to loop over all the boxes
font = cv.FONT_HERSHEY_PLAIN
colors= np.random.uniform(0,255,size= (len(boxes),4))
for i in range(len(boxes)):
    if i in indexes:
        x,y,ww,hh = boxes[i]
        label= str(classes[class_ids[i]])
        confi = str(round(confidences[i],2))
        color = colors[i]
        #cv.rectangle(img,(x,y),(x+ww,y+hh),color,2)
        #cv.putText(img,label,(x,y+20),font,5,(0,255,255),4)
        img_ = img[y:y+hh,x:x+ww]
        cv.imwrite(str(x)+".jpg" ,img_)

cv.imshow("Cropped", img)
cv.waitKey()
cv.destroyAllWindows() 