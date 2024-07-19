import numpy as np
import cv2
import math

srcArr_cameras = [
    np.float32([[798, 293], [1127, 293], [1881, 746], [881, 852]]),
    np.float32([[450, 424], [769, 413], [710, 660], [14, 703]]),
    np.float32([[1004, 338], [1326, 341], [1887, 740], [1195, 795]])
]

dstArr = np.float32([[0, 0], [500, 0], [500, 1800], [0, 1800]])

'''
transform_matrixs = []
for srcArr in srcArr_cameras:
    transform_matrixs.append(cv2.getPerspectiveTransform(srcArr, dstArr))
print(transform_matrixs)
'''

transform_matrixs = [
    np.float32([[-6.81700021e+00,  1.01218429e+00,  5.14339617e+03],
       [-3.86813270e-15, -5.50274840e+01,  1.61230528e+04],
       [ 8.92195004e-04, -2.21538905e-02,  1.00000000e+00]]),
    np.float32([[-1.95877711e+00, -3.06102802e+00,  2.17932558e+03],
       [-6.24164524e-01, -1.81007712e+01,  7.95560103e+03],
       [-4.65876860e-05, -5.19741222e-03,  1.00000000e+00]]),
    np.float32([[ 1.06871112e+01, -4.46660445e+00, -9.22014732e+03],
       [-6.50848671e-01,  6.98577573e+01, -2.29584699e+04],
       [-1.58995050e-03,  2.33547635e-02,  1.00000000e+00]])
]

def get_transformation_bound(camera_index):
    return srcArr_cameras[camera_index].astype(np.int32)

def test_in_transform_range(location):
    if cv2.pointPolygonTest(dstArr, location.tolist(), 1) > 0:
        return True
    else:
        return False

def location_transfomation(src, camera_index):
    src = np.float32(src)
    trans = cv2.perspectiveTransform(src.reshape(-1, 1, 2), transform_matrixs[camera_index])[0][0]
    trans = np.round(trans).astype(int)
    return trans

def zone_transfomation(zone, camera_index):
    trans_zone = []
    for location in zone:
        trans_zone.append(location_transfomation(location, camera_index))
    return np.array(trans_zone, dtype=np.int32)
