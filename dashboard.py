import cv2
import numpy as np
import math


def draw_corrected_angle(im0, corrected_angle, start_point):
    # 計算箭頭的終點
    arrow_len = 100
    rad = np.deg2rad(-1 * ((corrected_angle + 90) % 360))
    if math.isnan(rad):
        return im0
    end_point = (int(start_point[0] + arrow_len * np.cos(rad)), int(start_point[1] + arrow_len * np.sin(rad)))

    # 在原圖 im0 上畫箭頭
    im0 = cv2.arrowedLine(im0, start_point, end_point, (0, 0, 0), 4)

    return im0

def draw_obstacle(im0, obstacle_orientation):
    
    # 獲取圖像中心的坐標
    center_x, center_y = 150, 150

    # 在圖像中心畫一個白色圓圈
    circle_radius = 100  # 可以根據需要調整半徑
    circle_color = (255, 255, 255)  # BGR格式的白色
    cv2.circle(im0, (center_x, center_y), circle_radius, circle_color, -1)  # -1表示填充圓圈

    for angle in obstacle_orientation:
       
        # 繪製障礙物線
        rad = np.deg2rad(-1 * ((angle + 90) % 360))
        end_point = (int(center_x + 80 * np.cos(rad)), int(center_y + 80 * np.sin(rad)))
        cv2.circle(im0, end_point, 15, (0, 0, 255), -1)  # -1表示填充圓圈

    return im0
