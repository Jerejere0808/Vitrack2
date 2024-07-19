import numpy as np

def distance_point_to_segment(p, p1, p2):
    # 計算線段的向量
    v = p2 - p1
    
    # 計算點 P 到 P1 的向量
    w = p - p1
    
    # 計算投影向量的長度
    dot_product = np.dot(w, v)
    v_length = np.dot(v, v)
    
    # 計算投影比例
    t = dot_product / v_length
    
    # 如果 t 小於 0，最短距離是 P 到 P1 的距離
    if t < 0:
        distance = np.linalg.norm(p - p1)
    # 如果 t 大於 1，最短距離是 P 到 P2 的距離
    elif t > 1:
        distance = np.linalg.norm(p - p2)
    # 否則，最短距離是 P 到線段的垂直距離
    else:
        projection = p1 + t * v
        distance = np.linalg.norm(p - projection)
    
    return distance

def nearest_point_on_segment(p, p1, p2):
    v = p2 - p1
    w = p - p1
    dot_product = np.dot(w, v)
    v_length = np.dot(v, v)
    t = dot_product / v_length
    if t < 0:
        nearest_point = p1
    elif t > 1:
        nearest_point = p2
    else:
        nearest_point = p1 + t * v   
    return nearest_point

def nearest_point_on_polygon(p, poly):
    nearest_point = None
    min_distance = float("inf")
    for i in range(len(poly)):
        p1 = poly[i]
        p2 = poly[(i + 1) % len(poly)]
        tmp_nearest_point = nearest_point_on_segment(p, p1, p2)
        distance = np.linalg.norm(p - tmp_nearest_point)
        if distance < min_distance:
            min_distance = distance
            nearest_point = tmp_nearest_point
    return nearest_point