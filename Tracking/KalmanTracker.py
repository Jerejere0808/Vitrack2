from filterpy.kalman import KalmanFilter
import numpy as np
import lap

def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) 

def iou_batch(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)
    
def convert_bottonmid_wh_to_bbox(bottom_mid, w_and_h):

    return [bottom_mid[0] - w_and_h[0]/2.,bottom_mid[1] - w_and_h[1],bottom_mid[0] + w_and_h[0]/2.,bottom_mid[1]]

def convert_bbox_to_z(bbox):

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x,score=None):

    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
    

class KalmanBoxTracker(object):

  count = 0
  def __init__(self,bbox):

    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):

    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):

    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):

    return convert_x_to_bbox(self.kf.x)
  
class KalmanTrack(object):
    def __init__(self):
        self.trackers = []
        self.finded_users = dict()

        self.index2users = []
        self.users2index = dict()

        self.index2new_det = []

        self.trks = []
        self.dets = []

    def build_index2users(self, cur_finded_users):
        
        del_users = set(self.finded_users.keys()) - set(cur_finded_users.keys())
        for del_user in del_users:
            del_index = self.users2index[del_user]
            del self.index2users[del_index]
            del self.trackers[del_index]

        self.users2index.clear()
        for i in range(len(self.trackers)):
            self.users2index[self.index2users[i]] = i

        new_users = set(cur_finded_users.keys()) - set(self.finded_users.keys())
        for new_user in new_users:
            bottom_mid = [cur_finded_users[new_user][0], cur_finded_users[new_user][1]]
            w_and_h = [cur_finded_users[new_user][2], cur_finded_users[new_user][3]]
            self.trackers.append(KalmanBoxTracker(convert_bottonmid_wh_to_bbox(bottom_mid, w_and_h)[:]))
            self.index2users.append(new_user)
            self.users2index[new_user] = len(self.trackers)-1

        self.finded_users = cur_finded_users.copy()
    
    def build_index2new_det(self, new_det):
        self.index2new_det.clear()
        for i in range(len(new_det)):
            *xyxy, conf, cls, match_key = new_det[i]
            if(int(cls) == 0):
                self.index2new_det.append(i)
   
    def build_trks(self):
        self.trks = np.zeros((len(self.trackers), 5))
        for t, trk in enumerate(self.trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

    def build_dets(self, new_det):
        self.dets.clear()
        for index in self.index2new_det:
            *xyxy, conf, cls, match_key = new_det[index]
            self.dets.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], 0])
    
    def iou_batch(self, re_id_matched, detections, trackers):

        matched_det = [m[0] for m in re_id_matched]
        matched_users = [m[1] for m in re_id_matched]
   
        iou_matrix = np.zeros((len(self.index2new_det), len(self.index2users)))
        if len(self.index2new_det) != 0 and len(self.index2users) != 0:
            for i in range(len(self.index2new_det)):
                for j in range(len(self.index2users)):
                    #*xyxy, conf, cls, match_key = new_det[self.index2new_det[i]]
                    if i not in matched_det and j not in matched_users:
                        x1_box1, y1_box1, x2_box1, y2_box1 = detections[i][0], detections[i][1], detections[i][2], detections[i][3]
                        x1_box2, y1_box2, x2_box2, y2_box2 = trackers[j][0], trackers[j][1], trackers[j][2], trackers[j][3]
                        
                        # 計算 Intersection 的寬度和高度
                        w = min(x2_box1, x2_box2) - max(x1_box1, x1_box2)
                        h = min(y2_box1, y2_box2) - max(y1_box1, y1_box2)
                        
                        # 如果重疊區域為負或零，表示沒有重疊，IoU 為零
                        if w <= 0 or h <= 0:
                            iou_matrix[i][j] = 0
                        else:
                            # 計算 Intersection 和 Union 的面積
                            area_intersection = w * h
                            area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
                            area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
                            area_union = area_box1 + area_box2 - area_intersection

                            # 計算 IoU
                            iou = area_intersection / area_union
                            
                            iou_matrix[i][j] = iou
                    else:
                        iou_matrix[i][j] = 0.0
        return iou_matrix
    
    def associate_detections_to_trackers(self, re_id_matched, detections, trackers, iou_threshold = 0.05):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if(len(trackers)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

        #distance_matrix = iou_batch(detections, trackers)
        iou_matrix = self.iou_batch(re_id_matched, detections, trackers)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0,2))
        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)

        #filter out matched with low IOU
        matches = []

        for m in matched_indices:
            if(iou_matrix[m[0], m[1]]<iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def get_index2users(self):
        return self.index2users
    
    def get_index2new_det(self):
        return self.index2new_det

    def match(self, new_det, re_id_matched):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        
        # get predicted locations from existing trackers.
        self.build_trks()
        self.build_dets(new_det)

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(re_id_matched, self.dets, self.trks)

        # update matched trackers with assigned detections

        for m in re_id_matched:
            self.trackers[m[1]].update(self.dets[m[0]][:])

        for m in matched:
            self.trackers[m[1]].update(self.dets[m[0]][:])

        return matched
