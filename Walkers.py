import cv2
clf=cv2.CascadeClassifier("haarcascade_fullbody.xml")

cam=cv2.VideoCapture("walking.avi")
while True:
    ret,frame=cam.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=clf.detectMultiScale(grey,1.2,3)
    print((face))
    for x,y,w,h in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow("frame",frame)

    if cv2.waitKey(5)==32:
        break
cam.release()
cv2.destroyAllWindows()