import cv2
import numpy as np
import math
capture = cv2.VideoCapture(0)
capture1 = cv2.VideoCapture(1)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)
fps = capture.get(cv2.CAP_PROP_FPS)
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

writer = cv2.VideoWriter('./output.mp4', fmt, fps, size)


#createWindow

# 無処理
def nothing(x):
    pass

# ウィンドウの生成
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# トラックバーの生成
cv2.createTrackbar('minH', 'image', 0, 255, nothing)
cv2.createTrackbar('maxH', 'image', 255, 255, nothing)
cv2.createTrackbar('minS', 'image', 0, 255, nothing)
cv2.createTrackbar('maxS', 'image', 255, 255, nothing)
cv2.createTrackbar('minV', 'image', 0, 255, nothing)
cv2.createTrackbar('maxV', 'image', 255, 255, nothing)

while True:
    #inport
    ret,frame = capture.read()
    ret,frame1 = capture1.read()
    frame =cv2.flip(frame, -1)
    if ret == False:
        print('Can`t get data')
        break
    # トラックバーの値の取得
    minH = cv2.getTrackbarPos('minH', 'image')
    minS = cv2.getTrackbarPos('minS', 'image')
    minV = cv2.getTrackbarPos('minV', 'image')
    maxH = cv2.getTrackbarPos('maxH', 'image')
    maxS = cv2.getTrackbarPos('maxS', 'image')
    maxV = cv2.getTrackbarPos('maxV', 'image')
    
    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(frame_hsv, np.array([minH, minS, minV]), np.array([maxH, maxS, maxV]))
    
    #arm_serch
    kernel = np.ones((5,5),np.uint8)
    for i in range(10):
        mask = cv2.bilateralFilter(mask, d=3, sigmaColor=50, sigmaSpace=100)
    for i in range(20):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 輪郭抽出する。
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 面積が最大の輪郭を取得する
    if contours is not None:
        contour = max(contours, key=lambda x: cv2.contourArea(x))
    
    
    # マスク画像を作成する。
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
    mask1 = cv2.bitwise_not(mask)
    #3ch
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    mask1 = cv2.cvtColor(mask1,cv2.COLOR_GRAY2BGR)
    
    out = cv2.bitwise_and(frame,mask)
    out1 = cv2.bitwise_and(frame1,mask1)
    out = cv2.add(out1,out)
    
    cv2.line(out, (0, height//2), (width, height//2), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(out, (width//2, height//2), 5, (255, 255, 0), thickness=-1)
    
    cv2.imshow("Dst Image", out)
    writer.write(out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture.release()
writer.release()
cv2.destroyAllWindows()
