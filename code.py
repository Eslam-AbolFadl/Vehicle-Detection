from moviepy.editor import *
import numpy as np
import pandas as pd 
import os
import cv2
import glob
import time
import matplotlib.pyplot as plt



image2=cv2.imread(os.path.join("/home/saf/yolo-det/projenv/bin","test6.jpg"))


def imageadd( i,img1, img2 ): 
   # I want to put logo on top-left corner, So I create a ROI
   img1.setflags(write=1)
   rows,cols,channels = img2.shape
   roi = img1[0:rows, cols*2:cols*(3)]
   plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
   # Now create a mask of logo and create its inverse mask also
   img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
   ret,mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
   mask_inv = cv2.bitwise_not(mask)
   # Now black-out the area of logo in ROI
   img1_bg  = cv2.bitwise_and(roi,roi,mask = mask_inv)
   img1_bg=0
   # Take only region of logo from logo image.
   img2_fg  = cv2.bitwise_and(img2,img2,mask = mask)
   # Put logo in ROI and modify the main image
   dst     = cv2.add(img1_bg,img2_fg)
   img1[0:rows, cols*i:cols*(i+1) ] = dst
   print(i,"\n")
   return img1

def trackbar2(x):
    confidence = x/100
    r = r0.copy()
    for output in np.vstack(outputs):
        if output[4] > confidence:
            x, y, w, h = output[:4]
            p0 = int((x-w/2)*416), int((y-h/2)*416)
            p1 = int((x+w/2)*416), int((y+h/2)*416)
            cv2.rectangle(r, p0, p1, 1, 1)
         
        
def pre(image):        
        # Give the configuration and weight files for the model and load the network.
    
    weightspath=os.path.join("/home/saf/yolo-det/projenv/bin","yolov3.weights")
    configpath=os.path.join("/home/saf/yolo-det/projenv/bin","yolov3.cfg")
    net= cv2.dnn.readNetFromDarknet(configpath,weightspath)


    # determine the output layer

    layers=net.getLayerNames()
    outlayers= [layers[i - 1] for i in net.getUnconnectedOutLayers()]
    # Load names of classes and get random colors
    global classes
    classes= open('/home/saf/yolo-det/projenv/bin/coco_classes').read().strip().split('\n')
    np.random.seed(25)
    global colors 
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')


    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]
    my_project_dirblob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    #cv2.createTrackbar('confidence','window' 'blob', 50, 101, trackbar2)
    #trackbar2(50)
    #my_project_dir
    net.setInput(blob)
    t0 = time.time()
    global outputs 
    outputs = net.forward(outlayers)
    t = time.time()
    print('time=', t-t0)
        
def pipeline (image):
    pre(image)   


    
    
    boxes = []
    confidences = []
    classIDs = []
    h, w = image.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    l=0
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
        
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cropim= image[y:(y + h),x:x+w,0:3]
            image=imageadd(l,image,cropim)
            l=l+1
    return image           

#cv2.imshow('window', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



output = '/home/saf/yolo-det/projenv/bin/output_video01.mp4'
clip1 = VideoFileClip("/home/saf/yolo-det/projenv/bin/project_video_01_01.mp4")


out_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
%time out_clip.write_videofile(output, audio=False)

HTML("""
<video  width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output))
