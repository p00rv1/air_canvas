import mediapipe as mp
import time
import numpy as np
import cv2
import os
import handminmod as htm
folderPath="draw"
myList=os.listdir(folderPath)
#print(myList)
wCam,hCam=1280,720
cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0
erasThick=50
overlayList=[]
for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
#print(len(overlayList))
header=overlayList[2]
drawColor=(208, 160, 64)
detector=htm.HandDetector(detectionCon=0.85)
imgCanvas=np.zeros((720,1280,3),np.uint8)
brushThick=10
xp,yp=0,0
while True:
    success,img=cap.read()
    img=cv2.flip(img,1)
    img=detector.findHands(img,draw=False)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        #print(lmList)
        x1,y1=lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers=detector.fingersUp()
        #print(fingers)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img,(x1,y1-15),(x2,y2+15),drawColor,cv2.FILLED)
            print("Selection Mode")
            if y1<125:
                if 250<x1<450:
                    header=overlayList[2]
                    drawColor=(208, 160, 64)
                elif 550<x1<750:
                    header=overlayList[1]
                    drawColor = (0, 200, 0)
                elif 800<x1<950:
                    header=overlayList[0]
                    drawColor = (0, 0, 200)
                elif 1050 < x1 < 1200:
                    drawColor = (0, 0, 0)
                    header = overlayList[3]

        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            print("Draw Mode")
            if xp==0 and yp==0:
                xp,yp=x1,y1
            if drawColor==(0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, erasThick)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, erasThick)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThick)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThick)
            xp,yp=x1,y1
        if fingers[1] == False:
            xp, yp = 0, 0

    imgGray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv=cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img =cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgCanvas)
    img[0:125,0:1280]=header
    cv2.imshow("Image",img)
    #cv2.imshow("Im", imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
