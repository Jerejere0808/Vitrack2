import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms 
from numpy import random
import numpy as np
import os
import string
import threading
import math
import requests
import lap
from PIL import Image

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tactile_parameter import tactile_parameter
from handover_paramter import check_handover_imu, get_handover_bounds, check_yaw_range, handover_condition
from perspective_transformation import location_transfomation, zone_transfomation, test_in_transform_range, get_transformation_bound

# Re_id
from re_id.utils import load_network, fuse_all_conv_bn
from re_id.model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, ft_net_convnext, ft_net_cbam
from re_id.config import reid_cfg, update_reid_config
from re_id.Re_id_Matcher import Re_id_Match

# Tracking
from Tracking.KalmanTracker import KalmanTrack 
from Tracking.Re_idTracker import Re_idTrack

#IMU
from IMU.datasets import IMU_Dataset
from IMU.receiver import server_url

#deviation_correct
from deviation_correct import nearest_point_on_polygon

#show dashboard
from dashboard import draw_obstacle, draw_corrected_angle

inference = []
average_imu_change = []
frame_times = []
global average_frame_time 
deactivate_time = 0

def detect(reid_cfg, save_img=True):
    source, camera_num, weights, view_img, save_txt, imgsz, trace = opt.source, opt.camera_num,opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    average_frame_time = 0.15
    
    save_img = not opt.nosave # and not source.endswith('.txt')  # save inference images
    #save_img = False
    #view_img = False

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    vid_writers = [None] * camera_num
    if webcam:
        view_img = check_imshow()
        #view_img = False
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # ---------handover---------
    #應該要出現的users，但尚未發現
    finding_users = [{} for _ in range(camera_num)]

    #已經找到的users
    finded_users = [{} for _ in range(camera_num)]

    #待確認的handover使用者
    handover_dict = {}

    #已經判斷過yaw值確認是否要handover的user
    handover_list = []
    not_handover_list = []

    #users特徵
    user_features = {}
    user_features_fusion_confidence = {}
    # ---------handover---------

    # ---------Re_id---------
    print(reid_cfg)
    if reid_cfg.use_dense:
        model_structure = ft_net_dense(reid_cfg.nclasses, stride = reid_cfg.stride, linear_num=reid_cfg.linear_num)
    elif reid_cfg.use_NAS:
        model_structure = ft_net_NAS(reid_cfg.nclasses, linear_num=reid_cfg.linear_num)
    elif reid_cfg.use_swin:
        model_structure = ft_net_swin(reid_cfg.nclasses, linear_num=reid_cfg.linear_num)
    elif reid_cfg.use_swinv2:
        model_structure = ft_net_swinv2(reid_cfg.nclasses, (reid_cfg.h,reid_cfg.w),  linear_num=reid_cfg.linear_num)
    elif reid_cfg.use_convnext:
        model_structure = ft_net_convnext(reid_cfg.nclasses, linear_num=reid_cfg.linear_num)
    elif reid_cfg.use_efficient:
        model_structure = ft_net_efficient(reid_cfg.nclasses, linear_num=reid_cfg.linear_num)
    elif reid_cfg.use_hr:
        model_structure = ft_net_hr(reid_cfg.nclasses, linear_num=reid_cfg.linear_num)
    elif reid_cfg.ft_net_cbam:
        model_structure = ft_net_cbam(reid_cfg.nclasses, stride = reid_cfg.stride, ibn = reid_cfg.ibn, linear_num=reid_cfg.linear_num)
    else:
        model_structure = ft_net(reid_cfg.nclasses, stride = reid_cfg.stride, ibn = reid_cfg.ibn, linear_num=reid_cfg.linear_num)
    
    reid_model = load_network(model_structure, reid_cfg.name)
    reid_model.classifier.classifier = nn.Sequential()

    # Change to test mode
    reid_model = reid_model.eval()
    if reid_cfg.use_gpu:
        reid_model = reid_model.cuda()

    print('Here I fuse conv and bn for faster inference, and it does not work for transformers. Comment out this following line if you do not want to fuse conv&bn.')
    
    reid_model = fuse_all_conv_bn(reid_model)

    h, w = reid_cfg.h, reid_cfg.w
    data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((h, w), interpolation=3),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    Re_id_Matchs = [Re_id_Match() for _ in range(camera_num)]
    # ---------Re_id---------
        
    # ---------tracking---------
    KalmanTracks = [KalmanTrack() for _ in range(camera_num)]
    Re_idTracks = [Re_idTrack() for _ in range(camera_num)]
    # ---------tracking---------

    # ---------IMU---------
    fps_count = 0
    imu_dataset = IMU_Dataset("IMU\\IMU_DATA\\test4\\handover.csv")
    imu_t = time.time()
    imu_data = {"user1":0, "user2":0}
    north_ori = [340, 160, 340]
    # ---------IMU---------

    # ---------obstacle---------
    # 記錄畫面中所有滿足"危險"條件的物件 key為類別名稱 value為二維陣列 存放該類別物件(可能有多個)的x,y 
    obstacle_list = [{} for _ in range(camera_num)]
    #obstacle_index = [1, 2, 3, 5, 7, 13, 15, 16, 24, 25, 26, 28, 32, 39, 56, 57, 58, 75]
    obstacle_index = [1, 2, 5, 7, 56]
    # ---------obstacle---------

    t0 = time.time()
    for path, img, im0s, imu_dataset_id, vid_cap in dataset:
        
        fps_count = (fps_count + 1) % 1000
        start_time = time.time()

        # ---------IMU---------
        t, yaw = imu_dataset[imu_dataset_id]
        imu_data["user1"] = int(yaw)
        imu_data["user2"] = int(yaw)
        # ---------IMU--------- 

        # ---------handover---------
        #處理heandover
        for user in handover_dict:
            cur_camera, next_camera, imu_record = handover_dict[user]

            #滿幾筆IMU就判斷handover
            if len(imu_record) == 5:
                in_range = True
                for yaw in imu_record:
                    if check_yaw_range(yaw, cur_camera, next_camera) == False:
                        in_range = False
                        break
                if in_range:
                    handover_list.append(user)
                else:
                    not_handover_list.append(user)

            #每過幾幀就紀錄一筆IMU
            else:
                if(fps_count%12 == 0):
                    imu_record.append(imu_data[user])
                    handover_dict[user] = (cur_camera, next_camera, imu_record)

        if handover_list != []:
            for user in handover_list:
                #print("handover !!")
                del handover_dict[user]
                waiting_times = 200
                #把user加入下一個監視器的finding_users
                finding_users[next_camera][user] = [next_camera, cur_camera, waiting_times]
                dataset.activate_camera(next_camera)
                print("handover time: ", time.time() - deactivate_time)
                print("activate_camera: ", next_camera)
            handover_list = []

        if not_handover_list != []:
            for user in not_handover_list:
                del handover_dict[user]
                waiting_times = 200
                #把user加回原本監視器的finding_users
                finding_users[cur_camera][user] = [cur_camera, cur_camera, waiting_times]
                dataset.activate_camera(cur_camera)
                print("activate_camera: ", cur_camera)
            not_handover_list = []
        # ---------handover---------
            
        # ---------yolov7分類-------
            
        #只讓yolov7 判斷有接收串流的畫面
        #dataset有三個camera畫面，但只有activated_indexes有接收串流的畫面
        activated_indexes = dataset.activated_indexes()
        img = img[activated_indexes]
       
        if len(activated_indexes):
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # 根據"class"來排序 (讓"person"最後被處理)
            pred_index = pred[0][:,5].sort(descending = False)[1]
            pred[0] = pred[0][pred_index]
        
        # pred是一個list，每個element是一個tensor，代表每個activated cameras的畫面偵測結果 pred = [畫面0的偵測結果, 畫面1的偵測結果, ...]
        # ex: pred = [det0, det1, det2, ...]

        #每個det為pred的element，代表每個畫面中被偵測到的多個物件，det = [[物件1資訊], [物件2資訊], ...]
        # ex: det0 = [[x1, y1, x2, y2, conf, cls], [x1, y1, x2, y2, conf, cls], ...]

        #新建立一個new_pred，把沒有activated的camera的偵測結果補上空list
        #這樣使new_pred包含沒有activated的camera的偵測結果，並且保持new_pred的index和camera_index一致
        #ex: new_pred[0] 為 camera_index = 0 的偵測結果
        new_pred = [[] for _ in range(camera_num)]
        for i in range(len(activated_indexes)):
            activated_index = activated_indexes[i]
            new_pred[activated_index] = pred[i]

        # ---------yolov7分類-------

        # Process detections
        for camera_index, det in enumerate(new_pred):  # detections per image
            #if (dataset.camera_status(camera_index)):

            if webcam:  # batch_size >= 1   
                p, s, im0, frame = path[camera_index], '%g: ' % camera_index, im0s[camera_index].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            if (dataset.camera_status(camera_index)):

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # 原圖，避免受到疊圖影響
                img_ori = im0.copy() 
                # copy im0, and adjust weights between blind brick and im0
                overlay = im0.copy()
                # 重疊圖的比例 越高畫線的顏色越深
                alpha = 0.3
                
                #要判斷的區域
                transformation_bound = get_transformation_bound(camera_index)
                
                # 畫安全範圍
                cv2.drawContours(overlay, [tactile_parameter["safe_bound"][camera_index]], -1, (255, 0, 0), thickness=5)
                
                # 畫handover邊界
                handover_bounds = get_handover_bounds(camera_index)
                cv2.drawContours(overlay, handover_bounds, -1, (0, 0, 0), thickness=5)
                
                # Perform weighted addition of the input image and the overlay
                im0 = cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0)

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # ----------增加 match_key element----------

                    #把det的每個物件的資訊加上match_key element，這樣可以藉由match_key來判斷這個物件是不是同一個user
                    #原本det = [[x1, y1, x2, y2, conf, cls], [x1, y1, x2, y2, conf, cls], ...]
                    #new_det = [[x1, y1, x2, y2, conf, cls, match_key], [x1, y1, x2, y2, conf, cls, match_key], ...] 
                    new_det = det.cpu().tolist()
                    for d in new_det:
                        d.append(None)

                    # ----------增加 match_key element----------

                    # 用re id model取得pedestrian的feature，不是行人的feature為None
                    pedestrian_feature = [None for _ in range(len(new_det))]
                    for i in range(len(new_det)):
                        *xyxy, conf, cls, match_key = new_det[i]
                        if(int(cls) == 0):
                            cropobj = img_ori[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                            cropobj = cv2.cvtColor(cropobj.astype(np.uint8), cv2.COLOR_BGR2RGB)
                            reid_img = data_transforms(cropobj).unsqueeze(0).cuda()
                            feature = reid_model(reid_img)
                            pedestrian_feature[i] = feature
                    
                    # ----------行人持續追蹤----------


                    #建立index2users和index2new_det，為了在建立matrix並配對時，可以對應matrix的index到主程式的user和new_det的index
                    KalmanTracks[camera_index].build_index2users(finded_users[camera_index])
                    KalmanTracks[camera_index].build_index2new_det(new_det)
                    
                    #取得建好的index2users和index2new_det
                    index2users = KalmanTracks[camera_index].get_index2users()
                    index2new_det = KalmanTracks[camera_index].get_index2new_det()
                    
                    
                    #把index2users和index2new_det存進Re_idTrack，在建立配對matrix時,可以對應matrix的index到主程式的user和new_det的index
                    Re_idTracks[camera_index].load_index2users(index2users)
                    Re_idTracks[camera_index].load_index2new_det(index2new_det)
                    

                    
                    
                    #re id 配對(配對結果為[[pedestrian_index, user_index], [pedestrian_index, user_index], ...])
                    #pedestrian_index, user_index都是matrix的index，需透過index2users和index2new_det轉換成主程式的user和new_det的index
                    re_id_track_matched = Re_idTracks[camera_index].match(finded_users[camera_index], new_det, user_features, pedestrian_feature)
                    
                    for m in re_id_track_matched:
                        #把配對到的user match key存進new_det裡對應的element，代表這個新偵測到的new_det element是同一個user
                        new_det[index2new_det[m[0]]][6] = index2users[m[1]]
                        #如果是re id 配對到的話，可以融合feaure
                        user_features_fusion_confidence[index2users[m[1]]] = 0.5
                    
                    #kalman 配對(配對結果為[[pedestrian_index, user_index], [pedestrian_index, user_index], ...])
                    #pedestrian_index, user_index都是matrix的index，需透過index2users和index2new_det轉換成主程式的user和new_det的index
                    kalman_matched = KalmanTracks[camera_index].match(new_det, re_id_track_matched)
                    for m in kalman_matched:
                        #把配對到的user match key存進new_det裡對應的element，代表這個新偵測到的new_det element是同一個user
                        new_det[index2new_det[m[0]]][6] = index2users[m[1]]
                        #如果不是re id 配對到的話，不融合feaure(因為低score的re id可能不準確)
                        user_features_fusion_confidence[index2users[m[1]]] = 0.0
                    
                    # ----------行人持續追蹤----------
                    
                    # ----------行人重識別-----------
                    
                    Re_id_Matchs[camera_index].load_index2new_det(index2new_det)
                    re_id_matched = Re_id_Matchs[camera_index].match(finding_users[camera_index], new_det, user_features, pedestrian_feature, handover_bounds, transformation_bound)
                    
                    index2users = Re_id_Matchs[camera_index].get_index2users()
                    index2new_det = Re_id_Matchs[camera_index].get_index2new_det()
                    
                    for m in re_id_matched:
                        new_det[index2new_det[m[0]]][6] = index2users[m[1]]
                        #重識別直接取代上一個監視器的特徵
                        user_features_fusion_confidence[index2users[m[1]]] = 1.0

                    # ----------行人重識別----------
                        
                    # Write results
                    for *xyxy, conf, cls, match_key in reversed(new_det):

                        #紀錄時間
                        #time_cost = time.time()
                        #print(f"Time cost start")

                        # 標到人
                        if int(cls) == 0:
                            #print("find person!!")
                            
                            bottom_left = [int(xyxy[0]), int(xyxy[3])]
                            bottom_mid = [int((xyxy[0]+xyxy[2])/2), int(xyxy[3])]
                            bottom_right = [int(xyxy[2]), int(xyxy[3])]

                            w_and_h = [int((xyxy[2]-xyxy[0])/2), int((xyxy[3]-xyxy[1])/2)]
                            center_xy = [int(w_and_h[0]+xyxy[0]), int(w_and_h[1]+xyxy[1])]
                            
                            #沒有配對到的話，看是不是第一次進到系統(進到camera_index == 0)
                            #此部分省去BLE的部分，直接看是否在導盲磚的區塊上並當成使用者
                            #if(camera_index == 0 and match_key == None and not finded_users[camera_index]):
                            if(camera_index == 0 and match_key == None and len(finded_users[camera_index]) != 2):     
                                block_id = -2

                                #看位於導盲磚上的哪個區塊
                                for id, block_ in enumerate(tactile_parameter["blocks_bound"][camera_index]):
                                    if cv2.pointPolygonTest(block_, bottom_left, 1) > 0 \
                                        or cv2.pointPolygonTest(block_, bottom_mid, 1) > 0 \
                                        or cv2.pointPolygonTest(block_, bottom_right, 1) > 0:
                                        block_id = id
                                        break

                                #在導盲磚的前端出現的話當成使用者
                                if block_id == 0 and "user1" not in finded_users[camera_index]:
                                    match_key  = "user1"

                                block2 = np.array([[1188, 487], [1224, 618], [1564, 606], [1457, 450]], dtype=np.int32)
                                if cv2.pointPolygonTest(block2, bottom_left, 1) > 0 \
                                    or cv2.pointPolygonTest(block2, bottom_mid, 1) > 0 \
                                    or cv2.pointPolygonTest(block2, bottom_right, 1) > 0:
                                    match_key  = "user2"
                            
                            if match_key != None:

                                #IMU判斷人體方向
                                if match_key in imu_data:
                                    yaw = imu_data[match_key]
                                    orientation =  -1 * yaw if yaw < 0 else 360 - yaw
                                    orientation = int((orientation - north_ori[camera_index]) % 360)

                                #re id 取出目標特徵
                                cropobj = img_ori[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                                cropobj = cv2.cvtColor(cropobj.astype(np.uint8), cv2.COLOR_BGR2RGB)
                                
                                reid_img = data_transforms(cropobj).unsqueeze(0).cuda()
                                feature = reid_model(reid_img)

                                #如果是舊的使用者要融合舊特徵
                                if match_key in user_features and match_key in finded_users[camera_index] and match_key in user_features_fusion_confidence:
                                    ori_feature = user_features[match_key]
                                    fusion_confidence =  user_features_fusion_confidence[match_key]
                                    feature = torch.add(torch.mul(feature, fusion_confidence), torch.mul(ori_feature, 1 - fusion_confidence))
                                
                                user_features[match_key] = feature
                                
                                finded_users[camera_index][match_key] = [bottom_mid[0], bottom_mid[1], w_and_h[0], w_and_h[1], yaw, orientation]

                                #re id 配對到的話，把finding_users中的資料刪除                        
                                if match_key in finding_users[camera_index]:
                                    del finding_users[camera_index][match_key]

                                #User guiding system

                                #建立九宮格
                                grids = [0, 0, 0 ,0, 0, 0, 0, 0]
                                intervals = []
                                for i in range(8):
                                    start_angle = (45 * i - 22.5) % 360
                                    end_angle = (45 * i + 22.5) % 360
                                    intervals.append((start_angle, end_angle))
                                
                                #偏移矯正
                                user_location = [finded_users[camera_index][match_key][0],finded_users[camera_index][match_key][1]]
                                center_xy = location_transfomation(user_location, camera_index)

                                safe_bound = tactile_parameter["safe_bound"][camera_index]
                                safe_bound_xy = zone_transfomation(safe_bound, camera_index)

                                #超出安全範圍再矯正
                                if cv2.pointPolygonTest(safe_bound, user_location, 1) < 0:
                                    print(f"{match_key} is out of safe bound")

                                    nearest_xy = nearest_point_on_polygon(center_xy, safe_bound_xy)
                                    #cv2.circle(overlay, (int(nearest_point[0]), int(nearest_point[1])), 10, (0, 0, 255), 5)
                                    #im0 = cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0)
                                    
                                    orientation = finded_users[camera_index][match_key][5]

                                    #行人方向單位向量
                                    vx = round(math.cos((orientation + 90)%360 * (math.pi/180)), 2)
                                    vy = round(math.sin((orientation + 90)%360 * (math.pi/180)), 2)

                                    ux = nearest_xy[0] - center_xy[0]
                                    uy = -1 * (nearest_xy[1]-center_xy[1])

                                    #計算單位向量的分量
                                    magnitude = math.sqrt(ux**2 + uy**2)
                                    ux =  ux / magnitude
                                    uy =  uy / magnitude

                                    #行人方向和矯正方向的比重
                                    a, b = 1, 2
                                    
                                    #計算矯正角度
                                    corrected_angle = ((math.atan2(a * vy + b * uy, a * vx + b * ux)*180/math.pi) - 90)%360

                                    #根據使用者方向矯正行人角度
                                    #corrected_angle = (corrected_angle - orientation)%360
                                    for index, (start, end) in enumerate(intervals):
                                        if start < end:
                                            if start <= corrected_angle < end:
                                                grids[index] = 1
                                        else:
                                            if corrected_angle >= start or corrected_angle < end:
                                                grids[index] = 1

                                #障礙物判斷
                                distance = -1
                                ob_angle = -1
                                
                                obstacles_angle = []
                                for ob_name, ob_locations in obstacle_list[camera_index].items():
                                    
                                    for ob_xy in ob_locations:
                                        ob_xy = location_transfomation(ob_xy, camera_index)
                                        distance = sum([(a-b)**2 for a,b in zip(ob_xy, center_xy)])**0.5
                                        if test_in_transform_range(ob_xy) and distance < 500:
                                            dy = -1 * (ob_xy[1]-center_xy[1])
                                            dx = ob_xy[0] - center_xy[0]
                                            ob_angle = (math.atan2(dy, dx)*180/math.pi - 90)%360
                                            
                                            #根據使用者方向矯正障礙物角度
                                            ob_angle = (ob_angle - orientation)%360
                                            obstacles_angle.append(ob_angle)

                                for ob_angle in obstacles_angle:
                                    for index, (start, end) in enumerate(intervals):
                                        if start < end:
                                            if start <= ob_angle < end:
                                                grids[index] = 2
                                        else:
                                            if ob_angle >= start or ob_angle < end:
                                                grids[index] = 2
                                                    
                                # 顯示資訊的部分                    
                                #draw_obstacle(im0, obstacles_angle)
                                
                                #if cv2.pointPolygonTest(safe_bound, user_location, 1) < 0:
                                    #draw_corrected_angle(im0, corrected_angle, user_location)     

                                #確認handover
                                check_handover_imu([finded_users[camera_index][match_key][0], finded_users[camera_index][match_key][1]], finded_users[camera_index][match_key][4], camera_index, match_key, handover_dict)
                                if (match_key in handover_dict):
                                    del finded_users[camera_index][match_key]
                                    if((not finded_users[camera_index]) and (not finding_users[camera_index])):
                                        dataset.deactivate_camera(camera_index)
                                        deactivate_time = time.time()
                                    print("deactivate_camera: ", camera_index)
                                
                                # Print time (inference + NMS)
                                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                                inference.append(1E3 * (t2 - t1))
                                #print("average inference: ", sum(inference) / len(inference))
                            
                                end_time = time.time()
                                frame_times.append(end_time - start_time)
                                average_frame_time = sum(frame_times) / len(frame_times)
                                print("average frame time: ", average_frame_time)

                            else:
                                pass  

                        elif names[int(cls)] in [names[c] for c in obstacle_index]: 
                             
                            bottom_mid = [int((xyxy[0]+xyxy[2])/2), int(xyxy[3])]
        
                            # 尚未存進list
                            if names[int(cls)] not in obstacle_list[camera_index]:
                                obstacle_list[camera_index][names[int(cls)]] = [bottom_mid]
                            else: # 畫面中可能會有多個同樣類別的物件
                                new_ob = True
                                for location in obstacle_list[camera_index][names[int(cls)]]:
                                    # 如果距離太近代表是同一個物件，就不存進list
                                    if sum([(a-b)**2 for a,b in zip(location, bottom_mid)])**0.5 < 50:
                                        new_ob = False
                                        break
                                if new_ob:
                                    obstacle_list[camera_index][names[int(cls)]].append(bottom_mid)

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            if int(cls) == 0 and match_key != None:
                                #plot_one_box(xyxy, im0, label=f'ViTrack2 user {imu_data["user1"]} {orientation}', color=[0, 0, 255], line_thickness=3)
                                #plot_one_box(xyxy, im0, label=f'ViTrack2 user {match_key}', color=[0, 0, 255], line_thickness=3)
                                plot_one_box(xyxy, im0, label=f'{match_key}', color=[0, 0, 255], line_thickness=3)
                            elif int(cls) != 0:
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                            else:
                                pass
                        
                        #print("handover_dict: ", handover_dict)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path == None:  # new video
                        print("save_path: ", save_path)
                        print("vid_path: ", vid_path)
                        vid_path = save_path
                        if isinstance(vid_writers[camera_index], cv2.VideoWriter):
                            vid_writers[camera_index].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            print(f"video fps = {fps}")
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 20, im0.shape[1], im0.shape[0]
                            print(f"stream fps = {fps}")
                            save_path = save_path + '.mp4'
                            print(save_path)
                        for i in range(camera_num):
                            vid_writers[i] = cv2.VideoWriter(str(save_dir / p.name) + str(i) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writers[camera_index].write(im0)
                    if cv2.waitKey(1) == ord('q'):
                        for i in range(camera_num):
                            vid_writers[i].release()
                        break         

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done!!!! : ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--camera_num', type=int, default=1, help='number of sources')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    
    #re id
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--reid_name', default='ft_ResNet50', type=str, help='save model path')
    parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
    parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
    parser.add_argument('--use_efficient', action='store_true', help='use efficient-b4' )
    parser.add_argument('--use_hr', action='store_true', help='use hr18 net' )
    parser.add_argument('--ft_net_cbam', action='store_true')
    parser.add_argument('--PCB', action='store_true', help='use PCB' )
    parser.add_argument('--multi', action='store_true', help='use multiple query' )
    parser.add_argument('--fp16', action='store_true', help='use fp16.' )
    parser.add_argument('--ibn', action='store_true', help='use ibn.' )

    opt = parser.parse_args()
    print(opt)

    update_reid_config(reid_cfg, opt) #re_id

    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect(reid_cfg)