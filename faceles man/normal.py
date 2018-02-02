#run the code and remove your face from the front of webcam then enter any key 
import cv2,time
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
last_time=time.time()
ret,temp=cam.read()
temp=cv2.flip(temp,1)
avg1 = np.float32(temp)
avg2 = np.float32(temp)
 
a=input("Press any key")
while True:
    ret,img=cam.read()
    img=cv2.flip(img,1)
    print np.size(img,0),np.size(img,1)

    cv2.accumulateWeighted(img,avg2,0.01)

    res2 = cv2.convertScaleAbs(avg2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        faces = face_cascade.detectMultiScale(gray, 1.3, 1)
        #480 640
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
    ##    cv2.rectangle(img,(0,0),(640,y+h),(255,255,0),2)
        temp=res2[:y+h,:640]
        img[:y+h,:640]=temp
    except:
        continue
    cv2.putText(img,"Faceless man", (x+5,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("MY TRY",img)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    print "Frame rate == " ,1/(time.time()-last_time)
    last_time=time.time()
cam.release()
cv2.destroyAllWindows()
