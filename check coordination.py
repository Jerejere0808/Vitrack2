#import init_paths
import pandas as pd
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations

def DrawLine():
    # 開啟影片檔案
    cap = cv2.VideoCapture('./runs/detect/exp210/Produce_23.mp40.mp4')

    # 以迴圈從影片檔案讀取影格，並顯示出來
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        # 點擊滑鼠觸發function
        cv2.setMouseCallback('frame', onMouse)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       # draw circle here (etc...)
       print('x = %d, y = %d'%(x, y))

if __name__ == "__main__":
    DrawLine()