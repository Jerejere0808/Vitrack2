import pandas as pd
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations

points = []

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       # draw circle here (etc...)
       print('x = %d, y = %d'%(x, y))
       points.append((x, y))

def DrawLine():
    # 開啟影片檔案
    cap = cv2.VideoCapture('runs\detect\exp33\video0.mp40.mp4')
    # 以迴圈從影片檔案讀取影格，並顯示出來

    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        # 點擊滑鼠觸發function
        cv2.setMouseCallback('frame', onMouse)

        overlay = frame.copy()
        # 重疊圖的比例 越高畫線的顏色越深
        alpha = 0.3
        pts = np.array([[850, 290], [1185, 1070], [1520, 1070], [940, 290]], dtype=np.int32)
        #pts3 = np.array([[878, 291], [1338, 1062], [1414, 1062], [899, 291]], dtype=np.int32)
        cv2.drawContours(overlay, [pts], -1, (0, 0, 255), thickness=5)
        #cv2.drawContours(overlay, [pts2], -1, (0, 0, 0), thickness=5)
        
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        overlay = frame.copy()

        # Perform weighted addition of the input image and the overlay
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        for i in range(0, len(pts)):
            point = pts[i]
            cv2.circle(frame, point, 8, (0, 0, 255), -1)
                # Draw a red circle at the clicked position
            cv2.putText(frame, f'({point[0]}, {point[1]})', (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Display coordinates





        num = len(points)
        if num >= 2:
            for i in range(0, num):
                cv2.line(frame, points[i], points[(i + 1)% num], (0, 0, 255), 2)

        for i in range(0, num):
            point = points[i]
            cv2.circle(frame, point, 8, (0, 0, 255), -1)
              # Draw a red circle at the clicked position
            cv2.putText(frame, f'({point[0]}, {point[1]})', (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Display coordinates

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    DrawLine()