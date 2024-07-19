import numpy as np
import lap
import torch

class Re_idTrack(object):
    def __init__(self):

        self.index2users = []
        self.index2new_det = []
    
    def load_index2users(self, index2users):
        if index2users is not None:
            self.index2users = index2users.copy()
        else:
            self.index2users.clear()

    def load_index2new_det(self, index2new_det): 
        if index2new_det is not None:
            self.index2new_det = index2new_det.copy()
        else:
            self.index2new_det.clear()

    def get_index2users(self):
        return self.index2users
    
    def get_index2new_det(self):
        return self.index2new_det   

    def match(self, users, new_det, user_features, det_feature):

        if not self.index2users or not self.index2new_det:
            
            self.index2users.clear()
            for key, value in users.items():  
                self.index2users.append(key)

            self.index2new_det.clear()
            for i in range(len(new_det)):
                *xyxy, conf, cls, match_key = new_det[i]
                if(int(cls) == 0):
                    self.index2new_det.append(i)
            
        feature_distance_matrix = np.zeros((len(self.index2new_det), len(self.index2users)))
        matches = []

        if len(self.index2new_det) != 0 and len(self.index2users) != 0:

            threshold_xy = [100,100] # 水平(x軸)與垂直(y軸)的可能變動範圍
            for i in range(len(self.index2new_det)):
                for j in range(len(self.index2users)):

                    *xyxy, conf, cls, match_key = new_det[self.index2new_det[i]]
                    bottom_mid = [int((xyxy[0]+xyxy[2])/2), int(xyxy[3])]
                    
                    if conf > 0.88  and abs(users[self.index2users[j]][0]-bottom_mid[0]) <= threshold_xy[0] and abs(users[self.index2users[j]][1]-bottom_mid[1]) <= threshold_xy[1] :
                        euclidean_dist = torch.cdist(det_feature[self.index2new_det[i]], user_features[self.index2users[j]], p=2)[0][0]
                        feature_distance_matrix[i][j] = euclidean_dist
                    else:
                        feature_distance_matrix[i][j] = 16
            
            cost, x, y = lap.lapjv(feature_distance_matrix, extend_cost=True, cost_limit=15)
            matches = np.array([[y[i],i] for i in x if i >= 0])

        return matches

