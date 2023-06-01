import cv2
import numpy as np
##HAAR_FILE
HAAR_FILE = "./faced.xml"
##Cascadeに入力
cascade = cv2.CascadeClassifier(HAAR_FILE)

##カメラ入力、カメラ情報取得
capture = cv2.VideoCapture(0)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width,height)
fps = capture.get(cv2.CAP_PROP_FPS)

fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

#書き込み器作成
writer = cv2.VideoWriter('./output.mp4', fmt, fps/3, size)


while(True):
    ret,frame = capture.read()
    if ret == False:
        print('Can`t get data')
        break
    
    #Color2Gray
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    
    #3ch
    dstt = cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
    
    #FaceSerch
    face = cascade.detectMultiScale(frame)
    for x, y, w, h in face:
        if x != 0:
            cv2.putText(dstt, 'ON', (0, 80), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 3, cv2.LINE_AA)
        hsv = cv2.cvtColor(frame[y:h+y,x:x+w],cv2.COLOR_BGR2HSV)
        i=0
        j=0
        graycut = dstt[y:h+y,x:x+w]
        for i in range(w):
            for j in range(h):
                hc,sc,vc = hsv[j, i]
                if hc<100:
                    graycut[j,i]=frame[j+y,i+x]
        dstt[y:h+y,x:x+w] = graycut
        
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    out = dstt
    cv2.imshow('f', out)
    writer.write(out)

writer.release()
capture.release()
cv2.destroyAllWindows()