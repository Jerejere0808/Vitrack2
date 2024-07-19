import numpy as np
import cv2
from time import sleep

class SparseMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = {}

    def set_element(self, row, col, value):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.data[(row, col)] = value
        else:
            raise ValueError("Invalid row or column index")

    def get_element(self, row, col):
        if (row, col) in self.data:
            return self.data[(row, col)]
        else:
            return None

    def display(self):
        for i in range(self.rows):
            for j in range(self.cols):
                print(self.get_element(i, j))

def get_handover_bounds(cur_camera):
    bounds = []
    for j in range(handover_condition.cols):
        if handover_condition.get_element(cur_camera, j) is not None: 
            bounds.append(handover_condition.get_element(cur_camera, j)[0])
    return bounds

def check_yaw_range(yaw, cur_camera, next_camera):
    tactile_yaw = handover_condition.get_element(cur_camera, next_camera)[1]
    if tactile_yaw == 20:
        if -25 <= yaw <= 20 or 20 <= yaw <= 65:
            return True
        else:
            return False
    elif tactile_yaw == -160:
        if -160 <= yaw <= -115 or -180 <= yaw <= -160 or 155 <= yaw <= 180:
            return True
        else:
            return False
    return False
              
def check_handover_imu(location, yaw, cur_camera, match_key, handover_dict):
    if(match_key in handover_dict):
        return
    
    for next_camera in range(handover_condition.cols):
        if handover_condition.get_element(cur_camera, next_camera) is not None: 
            bound = handover_condition.get_element(cur_camera, next_camera)[0]
            if cv2.pointPolygonTest(bound, location, 1) > 0:
                next_camera = next_camera
                imu_record = []
                handover_dict[match_key] = (cur_camera, next_camera, imu_record)

global handover_condition
handover_condition = SparseMatrix(10, 10)
handover_condition.set_element(0, 1, (np.array([[824, 231], [849, 294], [1129, 295], [984, 226]], dtype=np.int32), 20))
handover_condition.set_element(1, 0, (np.array([[614, 325], [546, 362], [714, 372], [742, 324]], dtype=np.int32), -160))
handover_condition.set_element(1, 2, (np.array([[3, 672], [6, 1068], [658, 1067], [662, 1030], [96, 1007]], dtype=np.int32), 20))
handover_condition.set_element(2, 1, (np.array([[1134, 782], [1879, 591], [1919, 1074], [1149, 1077]], dtype=np.int32), -160))

