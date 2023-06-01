import cv2
import numpy as np

def serch(frame):
    for i in range(3):
        template = cv2.imread('./0'+ str(i+1) +'-2.png')
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        #template = cv2.blur(template, (5, 5))
        result = cv2.matchTemplate(frame,template,cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        print(f"max value: {maxVal}, position: {maxLoc},te"+str(i))
        # 類似度が 0.7 以上の位置を取得する。
        ys, xs = np.where(result >= 0.7)
        flag = 0
        for x, y in zip(xs, ys):
            flag = i+1     
        if flag == 1:
            power = 1
            bounce = 0
            sit = 0
            return power,bounce,sit
        elif flag == 2:
            power = 2
            bounce = 0
            sit = 0
            return power,bounce,sit
        elif flag == 3:
            power = 1
            bounce = 1
            sit = 1
            return power,bounce,sit
        else:
            power = -1
            bounce = -1
            sit = -1
    return power,bounce,sit

#画面のキャプチャを入力
capture = cv2.VideoCapture(0)


width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width,height)
fps = capture.get(cv2.CAP_PROP_FPS)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

writer = cv2.VideoWriter('./output.mp4', fmt,fps,size)

count = 0

while(True):
    count += 1
    #フレーム画像を取得
    ret,frame = capture.read()
    ret = True
    if ret == False:
        print('Can`t get data')
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    power,bounce,sit = serch(gray)
    dst = frame.copy()
    #条件分岐
    if sit == 0:
        sitcond = 'standing'
        cv2.putText(dst,
            text= 'power:'+str(power)+' bounce:'+ str(bounce)+' sitcond:'+ sitcond,
            org=(128, 72),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2.0,
            color=(255, 255, 0),
            thickness=4,
            lineType=cv2.LINE_4)
    elif sit ==  1:
        sitcond = 'sitting'
        cv2.putText(dst,
            text= 'power:'+str(power)+' bounce:'+ str(bounce)+' sitcond:'+ sitcond,
            org=(128, 72),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2.0,
            color=(255, 255, 0),
            thickness=4,
            lineType=cv2.LINE_4)
    else:
        sitcond = 'None'
        cv2.putText(dst,
            text= "Here isn't set position", 
            org=(128, 72),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2.0,
            color=(0, 255, 255),
            thickness=4,
            lineType=cv2.LINE_4)
       
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("img.jpg",dst)
        break
    if count == frame_count:
        break
    
    tmp = cv2.resize(dst, (1280, 720))
    cv2.imshow('f',tmp)
    writer.write(dst)

writer.release()
capture.release()
cv2.destroyAllWindows()