import cv2
import numpy as np
import math

capture = cv2.VideoCapture(1)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)
fps = capture.get(cv2.CAP_PROP_FPS)

before = None

fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

writer = cv2.VideoWriter('./output.mp4', fmt, fps, size)


while True:
    ret, frame = capture.read()
    # Color2Gray(defaault)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 動体検知
    if before is None:
        before = gray.astype("float")
        continue
    
    # 現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(gray, before, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(before))
    
    # frameDeltaの画像を２値化
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
    
    # 輪郭のデータを取得
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # モザイク処理を行うためのフレームを作成
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    
    #3ch
    dstt = cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
    
    mosaic_frame = dstt
    mosaic_frame = cv2.resize(mosaic_frame, (width//10, height//10), interpolation=cv2.INTER_LINEAR)
    mosaic_frame = cv2.resize(mosaic_frame, (width, height), interpolation=cv2.INTER_NEAREST)
    
    
    for target in contours:
        x, y, w, h = cv2.boundingRect(target)
        # 小さい変更点は無視
        if w < 80:
            continue 
        
        # 動体検知が行われた領域をモザイク処理から外す
        roi = frame[y:y+h, x:x+w]
        
        #マスク作成
        mask = np.zeros((h,w), np.float32)
        cv2.ellipse(mask,(w//2,h//2),(w//3,h//3),0,0,360,(255,255,255),thickness=-1)
        mask_b = cv2.GaussianBlur(mask,(int(w//4)*2+1,int(h//4)*2+1),w,h)
       
        
        mask_main = np.full((h,w,3), 255,np.float32)
        cv2.ellipse(mask_main,(w//2,h//2),(w//3,h//3),0,0,360,(0,0,0),thickness=-1)
        mask_main_b = cv2.GaussianBlur(mask_main,(int(w//4)*2+1,int(h//4)*2+1),w,h) 
        
        
        #マスク処理
        img1 = frame[y:y+h, x:x+w].astype(np.float32)
        mask_b_resized = cv2.cvtColor(mask_b, cv2.COLOR_GRAY2BGR)  # チャンネル数を一致させる
        mask_b_norm = mask_b_resized/255.0
        img1_masked = cv2.multiply(img1, mask_b_norm)
        cv2.imwrite("./img.jpg",img1_masked)
        
        img2 = mosaic_frame[y:y+h, x:x+w].astype(np.float32)
        mask_main_b_norm = mask_main_b/255.0
        img2_masked = cv2.multiply(img2, mask_main_b_norm)
        
        added = cv2.add(img1_masked, img2_masked,dtype=cv2.CV_8U)
        out = np.uint8(added)

        #合成
        mosaic_frame[y:y+h, x:x+w] = out
        
    # ウィンドウにフレームを表示
    cv2.imshow('frame', mosaic_frame)
    writer.write(mosaic_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
writer.release()
cv2.destroyAllWindows()
