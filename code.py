from moviepy.editor import *
import numpy as np
import pandas as pd 
import os
import cv2
import glob
import time
import matplotlib.pyplot as plt


 
def pipeline (image):
    image=cv2.resize(image,(416,416))
 
 
    weightspath=os.path.join("/home/saf/yolo-det/projenv/bin","yolov3.weights")
    configpath=os.path.join("/home/saf/yolo-det/projenv/bin","yolov3.cfg")
    net= cv2.dnn.readNetFromDarknet(configpath,weightspath)


    # determine the output layer

    layers=net.getLayerNames()
    outlayers= [layers[i - 1] for i in net.getUnconnectedOutLayers()]
    # Load names of classes 

    classes= open('/home/saf/yolo-det/projenv/bin/coco_classes').read().strip().split('\n')
#     np.random.seed(25)
#     global colors 
#     colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')


    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]
    net.setInput(blob)
    t0 = time.time()
    global outputs 
    outputs = net.forward(outlayers)
    t = time.time()
    print('time=', t-t0)

    
    
    boxes = []
    confidences = []
    classIDs = []
    h, w = image.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.7:
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
        
            #color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), (10,10,255), 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,10,255), 1)
            
            
            # extract detected vehicles and instert them on the top of the image
            cropim= image[y:(y + h),x:x+w,0:3]
            rows,cols,channels = cropim.shape          
            image[0:rows, cols*(2*l):cols*(2*l+1) ] = cropim
            l=l+1
    return image           





output = '/home/saf/yolo-det/projenv/bin/output_video01.mp4'  # change the path according to your own vidoe location on your PC
clip1 = VideoFileClip("/home/saf/yolo-det/projenv/bin/project_video_01_01.mp4")


out_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
%time out_clip.write_videofile(output, audio=False)

HTML("""
<video  width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output))
