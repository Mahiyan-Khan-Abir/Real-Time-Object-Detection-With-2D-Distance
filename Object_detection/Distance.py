import cv2
import numpy as np 

cap = cv2.VideoCapture(0)

_, prev = cap.read()
prev = cv2.flip(prev, 1)
_, new = cap.read()
new = cv2.flip(new, 1)

thres = 0.45

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    #success,img = cap.read()
    classIds, confs, bbox = net.detect(prev,confThreshold=thres)
    #print(classIds,bbox)                    
    diff = cv2.absdiff(prev, new)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.blur(diff, (5,5))
    _,thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    threh = cv2.dilate(thresh, None, 3)
    thresh = cv2.erode(thresh, np.ones((4,4)), 1)
    contor,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.circle(prev, (340,300), 4, (0,0,255), -1)
    cv2.line(prev, (0,300), (680,300 ), (0,255,0), 1)
    for contors in contor:				
            if cv2.contourArea(contors) > 1000:
                (x,y,w,h) = cv2.boundingRect(contors)
                (x1,y1),rad = cv2.minEnclosingCircle(contors)
                x1 = int(x1)
                y1 = int(y1)
                cv2.line(prev, (x1,300), (340, y1), (0,0,255), 1)
                cv2.line(prev, (x1,y1), (340, 300), (0,0,255), 1)
                cv2.putText(prev, "{}".format(int(np.sqrt((x1 - 340)**2 + (y1 - 300)**2))), (45,45),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
                #cv2.rectangle(prev, (x,y), (x+w,y+h), (0,255,0), 1)
                cv2.circle(prev, (x1,y1), 5, (0,255,255), -1)
                #a = (int(np.sqrt((x1 - 340)**2 + (y1 - 200)**2)))
                cv2.line(prev, (x1,300), (x1, y1), (255,255,0), 1)
                cv2.line(prev, (340,y1), (x1, y1), (255,255,0), 1)
                cv2.circle(prev, (x1,300), 5, (0,255,255), -1)
                cv2.circle(prev, (340,y1), 5, (0,255,255), -1)
                #print(a)
                #print(x1)
                #print(y1)
    if len(classIds) != 0:
        #print (classIds)
        #print(classNames)
                for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                        cv2.rectangle(prev,box,color=(0,255,0),thickness=1)
                        cv2.putText(prev,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
                        #cv2.putText(prev,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        #cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            #print(classNames[classId-1].upper())
					
	
    cv2.imshow("orig", prev)
	
    prev = new
    _, new = cap.read()
    new = cv2.flip(new, 1)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()