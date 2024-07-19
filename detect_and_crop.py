#Object Crop Using YOLOv7
import argparse
import time
import math
from pathlib import Path
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms 

from collections import deque
from numpy import random
from scipy.spatial import distance
from datetime import datetime
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
# BLE
from BLE.lib.models import get_model
from BLE.lib.config import cfg, update_config
from BLE.lib.utils import save_checkpoint, get_optimizer, create_logger,\
    check_side, check_in_range, check_in_range_eq, get_acceptable_range, get_corrected_direction, ParticleFilter, ParticleFilter_v2
from BLE.lib.datasets import RSSI_Dataset, RSSI_DatasetForTest
#MEBOW
from MEBOW.lib.models import get_pose_net


def detect(cfg, save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'crops').mkdir(parents=True, exist_ok=True) # make dir for cropped images
    crop_cnt = 0

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # ----------BLE----------
    # RSSI Matching model & Dataset
    rssi_model = get_model(cfg, False)
    rssi_dataset = RSSI_DatasetForTest(cfg)

    # Load Checkpoint
    checkpoint_file = os.path.join(cfg.MODEL.PRETRAINED)
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        rssi_model.load_state_dict(checkpoint['best_state_dict'])

    rssi_model = torch.nn.DataParallel(rssi_model).cuda()
    # testing
    rssi_model.eval()
    pf = ParticleFilter_v2(cfg)

    # 邊界範圍(看bottom mid xy)
    pts = np.array([[850, 290], [1185, 1070], [1520, 1070], [940, 290]], dtype=np.int32)
    # 下半身位於螢幕外的容許範圍(看center xy)
    pts2 = np.array([[1332, 800], [1525, 1070], [1766, 1070], [1540, 800]], dtype=np.int32)
    # 每個Block範圍 (Block 0~7分別對應從螢幕下方往上的格子)
    pts_blocks = np.array([
                        [[1113, 894], [1186, 1067], [1513, 1067], [1386, 894]],
                        [[1021, 680], [1113, 894], [1386, 894], [1228, 680]],
                        [[966, 554], [1017, 672], [1219, 672], [1134, 554]],
                        [[927, 463], [966, 554], [1134, 554], [1065, 463]],
                        [[898, 393], [927, 463], [1065, 463], [1013, 393]],
                        [[887, 366], [898, 393], [1013, 393], [993, 366]],
                        [[872, 332], [887, 366], [993, 366], [968, 332]],
                        [[856, 294], [872, 332], [968, 332], [938, 294]],], dtype=np.int32)

    # 導盲磚的線段頭尾兩點(直線)
    point1 = [891, 292]
    point2 = [1383, 1079]
    # 導盲磚範圍
    pts3 = np.array([[870, 291], [1330, 1070], [1422, 1070], [905, 291]], dtype=np.int32)

    # For obstacle detection
    obstacle_index = [1, 2, 3, 5, 7, 13, 15, 16, 24, 25, 26, 28, 32, 39, 56, 57, 58, 75]
    # class names
    # names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    #         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    #         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    #         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    #         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    #         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    #         'hair drier', 'toothbrush' ]

    # ----------BLE----------
    # ------偏離角度測試------
    # bias_df = pd.read_excel('0722test2_5.xlsx')
    # bias_df['prediction'] = 0
    # bias_df['error'] = 0
    # ---------MEBOW---------
    # Load MEBOW Model
    mebow_model = get_pose_net(cfg, False)
    
    # Load Pretrained Weights
    checkpoint_file2 = os.path.join(cfg.MODEL.MEBOW_PRETRAINED)
    if os.path.exists(checkpoint_file2):
        checkpoint2 = torch.load(checkpoint_file2)
        mebow_model.load_state_dict(checkpoint2)
    mebow_model = torch.nn.DataParallel(mebow_model).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # Data Preprocessing
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize([256,192]), normalize]
    )
    # testing
    mebow_model.eval()
    # ---------MEBOW---------

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
    if webcam:
        view_img = check_imshow()
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

    t0 = time.time()
    
    with torch.no_grad():
        # 每個frame都做一次rssi matching，但30的倍數時才更新data_id
        fps_count = 0
        block_id = 0 
        # rssi dataset counter
        data_id = 0
        # 追蹤列表，紀錄用戶上一次的位置，key為UUID，value為[center x,center y, orientation, leave_counter] 
        last_xy = {}
        # 每5個frame處理一次修正方向判斷，取出現次數最多的為結果。
        corrected_list = deque()
        for path, img, im0s, vid_cap in dataset:
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
            pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            #print(pred)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            find_target = False

            # 根據"class"來排序 (讓"person"最後被處理)
            pred_index = pred[0][:,5].sort(descending = False)[1]
            pred[0] = pred[0][pred_index]
            # 記錄畫面中所有滿足"危險"條件的物件 key為類別名稱 value為二維陣列 存放該類別物件(可能有多個)的x,y
            obstacle_list = {} 
            for i, det in enumerate(pred):  # detections per image

                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # ----------BLE----------
                # 原圖，避免受到疊圖影響
                img_ori = im0.copy() 
                # copy im0, and adjust weights between blind brick and im0
                overlay = im0.copy()
                # 重疊圖的比例 越高畫線的顏色越深
                alpha = 0.3
                # draw the blind brick range
                cv2.drawContours(overlay, [pts], -1, (0, 0, 0), thickness=5)
                # Perform weighted addition of the input image and the overlay
                im0 = cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0)
                # ----------BLE----------

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # 標到人
                        if int(cls) == 0:
                            # ----------BLE----------
                            tdd0 = time.time()
                            bottom_left = [int(xyxy[0]), int(xyxy[3])]
                            bottom_mid = [int((xyxy[0]+xyxy[2])/2), int(xyxy[3])]
                            bottom_right = [int(xyxy[2]), int(xyxy[3])]

                            w_and_h = [int((xyxy[2]-xyxy[0])/2), int((xyxy[3]-xyxy[1])/2)]
                            center_xy = [int(w_and_h[0]+xyxy[0]), int(w_and_h[1]+xyxy[1])]
                            # Part 1 - 用戶目標配對 
                            # 1.跟追蹤列表中的位置做比對
                            # UUID需要對應(尚未完成)
                            match_key = "" 
                            threshold_xy = [30,30] # 水平(x軸)與垂直(y軸)的可能變動範圍
                            for key, value in last_xy.items():
                                if abs(value[0]-center_xy[0]) <= threshold_xy[0] and abs(value[1]-center_xy[1]) <= threshold_xy[1] :
                                    match_key = key
                                break

                            # 如果框出的人沒配對到且在邊界範圍外(不在灰色範圍內且中心座標不位於容許範圍pts2中)則忽略
                            if not match_key and cv2.pointPolygonTest(pts, bottom_mid, 1) <= 0 and cv2.pointPolygonTest(pts2, center_xy, 1) <= 0:                        
                                continue                            

                            # 2.確認yolo框出的x,y屬於哪個Block
                            yolo_xyid = -2
                            for id, block_ in enumerate(pts_blocks):
                                if cv2.pointPolygonTest(block_, bottom_left, 1) > 0 \
                                    or cv2.pointPolygonTest(block_, bottom_mid, 1) > 0 \
                                    or cv2.pointPolygonTest(block_, bottom_right, 1) > 0:
                                    yolo_xyid = id
                                    break
                            # 下半身位於螢幕外的容許範圍的情況，當作在Block 0
                            if cv2.pointPolygonTest(pts2, center_xy, 1) > 0:
                                yolo_xyid = 0

                            # 若沒對應到追蹤列表中的位置，且框出的位置不在頭尾，表示只是在範圍內的路人，便忽略該目標
                            # 這部分之後可以優化看看
                            # if not match_key and yolo_xyid != 0 and yolo_xyid != 7:     
                            #     continue

                            # 若FPS為30 則每30幀更新一次data_id, 且UUID必須對應到用戶(尚未完成)
                            x = rssi_dataset[data_id]
                            if cfg.MODEL.TYPE == "DNN":
                                x = x.view(1, -1)
                            elif cfg.MODEL.TYPE == "LSTM" or cfg.MODEL.TYPE ==  "1DCNN":
                                x = x.view(1, x.size()[0], x.size()[1])

                            # 3.取得RSSI Matching結果
                            y_test = rssi_model(x)
                            # print(x)
                            # print(f"id:{data_id}")
                            # get top-k result
                            _, maxk = torch.topk(y_test, 1, dim=-1)
                            # run particle filter
                            maxk = pf.run(y_test)
                            block_id = maxk[0][0].item()

                            # 4.判斷yolo框出的x,y與RSSI matching的x,y是否差距過大(第一次的匹配)
                            #   或是 跟追蹤列表中的記錄有對應 (匹配後的持續追蹤)
                            if abs(yolo_xyid-block_id) < 2 or match_key :
                                # 5.MEBOW判斷人體方向
                                cropobj = img_ori[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                                crop_img = transform(cropobj)
                                crop_img = crop_img[np.newaxis,:,:,:]

                                output, hoe = mebow_model(crop_img)
                                orientation = hoe.detach().cpu().numpy()
                                orientation = 5*np.argmax(orientation)
                                # 判斷用戶是否已經離開的計數
                                leave_count = 0
                                # 更新用於追蹤的用戶位置 角度(center x, center y, direction)
                                if match_key not in last_xy:
                                    # 從最靠近攝影機的block進入，確保MEBOW的角度結果符合導盲磚方向
                                    # if yolo_xyid == 0:
                                    #     orientation = 15
                                    # # 從最遠離攝影機的block進入，確保MEBOW的角度結果符合導盲磚方向
                                    # elif yolo_xyid == 7:
                                    #     orientation = 180

                                    last_xy[cfg.UUID] = [center_xy[0], center_xy[1], w_and_h[0], w_and_h[1], orientation, leave_count]
                                # 匹配後的持續追蹤
                                else:
                                    # 可容許的角度變化閾值，避免預測的角度與實際落差太大
                                    angle_threshold = 30
                                    last_o = last_xy[match_key][4]
                                    angle_diff = abs(last_o - orientation)
                                    # 若角度變動超出範圍，則採用上次的角度，不進行更新
                                    if angle_diff >= angle_threshold and 360-angle_diff >= angle_threshold:
                                        orientation = last_o

                                    last_xy[match_key] = [center_xy[0], center_xy[1], w_and_h[0], w_and_h[1], orientation, leave_count]

                                # Part 2 - 判斷偏離與方向校正
                                # 計算身體方向角度與Ground True 差異
                                # bias_df['prediction'][data_id] = orientation
                                # bias_df['error'][data_id] = min(abs(bias_df['degree'][data_id] - orientation), 360-abs(bias_df['degree'][data_id] - orientation))
                                # 導盲磚線段在畫面上的角度[0,180)，垂直為0，水平則為90，逆時針增加。
                                theta_tp = 0
                                # 根據導盲磚分段的傾斜程度(對於視角而言)，稍微再調整theta tp
                                inclination = [0, 10, 20] # 越靠近鏡頭的部分看起來會越傾斜
                                inclination_threshold = [419,673,1080] # y軸數值
                                for index, y in enumerate(inclination_threshold):
                                    if center_xy[1] < y:
                                        theta_tp += inclination[index]
                                        # orientation += inclination[index]
                                        orientation %= 360
                                        break      

                                # acceptable range
                                ac_range = [0, 180]

                                # 計算用戶在導盲磚的左側(-1)or右側(+1)
                                side = check_side(point1, point2, bottom_mid)  

                                # Situation 1 (在導盲磚範圍內)
                                if cv2.pointPolygonTest(pts3, bottom_mid, 1) >= 0:
                                    deviate = False

                                # Situation 2 (在導盲磚範圍外，並介於導盲磚與邊界之間)
                                if cv2.pointPolygonTest(pts3, bottom_mid, 1) < 0 and cv2.pointPolygonTest(pts, bottom_mid, 1) >= 0:
                                    # 取得修正後的acceptable range
                                    ac_range = get_acceptable_range(ac_range, side, theta_tp)
                                    # check if the user's orientation is in acceptable range 
                                    if not check_in_range_eq(ac_range[0], ac_range[1], orientation): 
                                        deviate = True
                                    else:
                                        deviate = False

                                # Situation 3 (在邊界範圍外)
                                if cv2.pointPolygonTest(pts, bottom_mid, 1) < 0:
                                    deviate = True
                                    # 取得修正後的acceptable range
                                    ac_range = get_acceptable_range(ac_range, side, theta_tp)
                                    # print(f"head : {ac_range[0]} , tail : {ac_range[1]}")
                                    # print('---')
                                    # 調整theta start, theta end(用於後續提醒用戶修正方向)
                                    add_s = 20
                                    add_e = -20
                                    ac_range[0] = (ac_range[0] + add_s)%360
                                    ac_range[1] = (ac_range[1] + add_e)%360
                                    # # 取得修正後的acceptable range
                                    # ac_range = get_acceptable_range(ac_range, side, theta_tp)
                                    # print(f"head : {ac_range[0]} , tail : {ac_range[1]}")

                                corrected_direction = 0
                                # 偵測到偏離，判斷往左or往右
                                if deviate :
                                    corrected_direction, corrected_angle = get_corrected_direction(ac_range, orientation)
                                # print(f"orientation : {orientation}")
                                # print(f"ac_range : {ac_range[0]}, {ac_range[1]}")
                                # print(f"corrected: {corrected_direction}")
                                # Part3 － 障礙/危險物件感測
                                obstacle_check = 0
                                # 定義感測邊界
                                x_ratio = 4.
                                y_ratio = 2.
                                x_l = center_xy[0]-x_ratio*w_and_h[0] if center_xy[0]-x_ratio*w_and_h[0] >= 0 else 0
                                x_r = center_xy[0]+x_ratio*w_and_h[0] if center_xy[0]+x_ratio*w_and_h[0] <= im0.shape[1] else im0.shape[1]
                                y_l = center_xy[1]-y_ratio*w_and_h[1] if center_xy[1]-y_ratio*w_and_h[1] >= 0 else 0
                                y_r = center_xy[1]+y_ratio*w_and_h[1] if center_xy[1]+y_ratio*w_and_h[1] <= im0.shape[0] else im0.shape[0]
                                # 感測範圍
                                sensing_range = np.array([[x_l, y_l], [x_l, y_r], [x_r, y_r], [x_r, y_l]], dtype=np.int32)
                                # 視覺化顯示感測邊界
                                # overlay = im0.copy()
                                # cv2.drawContours(overlay, [sensing_range], -1, (125, 255, 125), thickness=-1)
                                # im0 = cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0)

                                danger = [0 for _ in range(8)]
                                # 8個方向可能性，基準0+-25度為往前，45+-25度為左前，90+-25度為往左，以此類推。
                                danger_direction = [0, 45, 90, 135, 180, 225, 270, 315]
                                # 導盲磚範圍內的正前方是否有障礙物
                                ob_ahead = False
                                # 判斷障礙/危險物件是否位於感測範圍中，並記錄在哪個相對方向
                                for obstacles in obstacle_list.values():
                                    for ob_xy in obstacles:
                                        distance = sum([(a-b)**2 for a,b in zip(ob_xy, center_xy)])**0.5
                                        if cv2.pointPolygonTest(sensing_range, ob_xy, 1) >= 0 and distance <= 150:
                                            if cv2.pointPolygonTest(pts, ob_xy, 1) >= 0:
                                                ob_ahead = True
                                            dy = -1 * (ob_xy[1]-center_xy[1])
                                            dx = ob_xy[0] - center_xy[0]
                                            ob_angle = (math.atan2(dy, dx)*180/math.pi)%360
                                            # 扣除90度是因為我們的座標體系N的方向為0度，逆時針增加，然而日常中的極座標系N的方向為90度
                                            # E的方向為0度，因此透過atan2()求出的角度還要進行旋轉修正(坐標軸逆時針旋轉90度)
                                            # 最後扣除用戶的朝向角度，得出障礙物與人物之間的角度差值(同樣方向為逆時針)
                                            theta_rotate = 90
                                            ob_angle = (ob_angle - theta_rotate - orientation)%360
                                            print(f"angle : {ob_angle}")
                                            # 確認障礙/危險物件在人的哪個相對方向
                                            for d_index, angle in enumerate(danger_direction):
                                                if check_in_range_eq(angle-22.5, angle+22.5, ob_angle):
                                                    danger[d_index] = 1
                                                    print(f"Obstacle in {d_index}")
                                                    # 不理會用戶背後的障礙物
                                                    if d_index not in [3,4,5]:
                                                        obstacle_check = 1
                                                    break

                                # Obstacle Avoidance
                                # 前方、左前、右前任一方向有障礙物
                                # if danger[0] or danger[1] or danger[-1]:
                                #     corrected_direction = 0
                                # 若偏移校正的方向為-1，也就是向右前方走，則確認用戶右方是否有障礙物
                                # 計算後的值可能為0 or -1，表示前方有障礙物或是向右行走來迴避，反之亦然。


                                # if corrected_direction == -1:
                                #     corrected_direction += (danger[-1] + danger[-2]) + d_0*danger[0]
                                # elif corrected_direction == 1:
                                #     corrected_direction += -(danger[1] + danger[2]) + d_0*danger[0]
                                # else:
                                #     corrected_direction += d_0*danger[0]

                                # 若正前方有障礙物，則必須事先決定哪個方向可以走
                                # 以電機系館1F外為例，左側為水溝蓋，因此希望用戶往人行道那側走
                                d_0 = 1 # -1:背對攝影機 1:面對攝影機
                                if danger[0] and ob_ahead:
                                    # 先確認用戶的朝向，若用戶從鏡頭近處走向遠處，則往右走。
                                    # 否則往左走，表示用戶從鏡頭遠處走向近處
                                    up_range = [270, 90] # MEBOW座標軸中從270度到90度的區間
                                    d_0 = -1 if check_in_range_eq(up_range[0], up_range[1], orientation) else 1 

                                corrected_direction += (danger[-1] + danger[-2]) + -1*(danger[1] + danger[2]) + d_0*danger[0]

                                corrected_direction = 1 if corrected_direction > 1 else corrected_direction
                                corrected_direction = -1 if corrected_direction < -1 else corrected_direction
                                # corrected_dict = {-1:'Rightward', 1:'Leftward', 0:'Maintain'}
                                # 紀錄10筆修正方向，取出現次數最多的方向作為最終結果
                                if len(corrected_list) == 10:
                                    corrected_list.popleft()
                                corrected_list.append(corrected_direction)
                                # print(f"direction :{corrected_direction}")
                                print(f"time cost : {time.time()-tdd0}")
                                find_target = True
                            else:
                                # find_target = True if find_target else find_target
                                find_target = False
                                continue
                        # 如果框到的物件類別屬於障礙物列表中，且有紀錄用戶上個時間點的位置資訊時，才將紀錄該物件的資訊(畫面中沒人則不管)
                        elif names[int(cls)] in [names[c] for c in obstacle_index] and sum([len(i) for i in last_xy]):

                            w_and_h = [int((xyxy[2]-xyxy[0])/2), int((xyxy[3]-xyxy[1])/2)]
                            center_xy = [int(w_and_h[0]+xyxy[0]), int(w_and_h[1]+xyxy[1])]
                            # 尚未存進list
                            if names[int(cls)] not in obstacle_list:
                                obstacle_list[names[int(cls)]] = [center_xy]
                            else: # 畫面中可能會有多個同樣類別的物件
                                obstacle_list[names[int(cls)]].append(center_xy)

                            
                        # ----------BLE----------                        
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')


                        if save_img or view_img:  # Add bbox to image
                            #label = f'{names[int(cls)]} {conf:.2f}'
                            label = f'{names[int(cls)]}'

                            # 顯示資訊的部分
                            overlay = im0.copy()
                            cv2.drawContours(overlay, [np.array([[0, 0], [0, 400], [330, 400], [330, 0]])], -1, (255, 255, 255), thickness=-1)
                            
                            cv2.putText(overlay, "0 / 360", (130, 27), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 0, 0), 1, cv2.LINE_AA)
                            cv2.putText(overlay, "90", (20, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 0, 0), 1, cv2.LINE_AA)
                            cv2.putText(overlay, "270", (250, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 0, 0), 1, cv2.LINE_AA)
                            cv2.putText(overlay, "180", (130, 295), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 0, 0), 1, cv2.LINE_AA)

                            if find_target:
                                # Orientation文字
                                text = "Orientation: " + str(orientation)
                                cv2.putText(overlay, text, (45, 320), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1, cv2.LINE_AA)
                                

                                # Orientation畫箭頭                              
                                start_point = (150, 150)
                                arrow_len = 100
                                rad = np.deg2rad(-1*((orientation + 90)%360))
                                end_point = (int(start_point[0] + arrow_len * np.cos(rad)), int(start_point[1] + arrow_len * np.sin(rad)))

                                overlay = cv2.arrowedLine(overlay, start_point, end_point, (0, 0, 0) , 4)
                                # cv2.putText(overlay, "0 / 360", (130, 27), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                #     1, (0, 0, 0), 1, cv2.LINE_AA)
                                # cv2.putText(overlay, "90", (20, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                #     1, (0, 0, 0), 1, cv2.LINE_AA)
                                # cv2.putText(overlay, "270", (250, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                #     1, (0, 0, 0), 1, cv2.LINE_AA)
                                # cv2.putText(overlay, "180", (130, 295), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                #     1, (0, 0, 0), 1, cv2.LINE_AA)

                                #方向調整文字
                                if fps_count % 10 == 0:
                                    final_direction = max(corrected_list , key=corrected_list.count)
                                # print(corrected_list)
                                # print(corrected_direction)
                                corrected_dict = {-1:'Walk right', 1:'Walk left', 0:'Go ahead'}
                                if len(corrected_list) == 10:
                                    text = corrected_dict[final_direction]
                                    text = text + ', Degree: ' + str(corrected_angle) if final_direction else text
                                    cv2.putText(overlay, text, (45, 350), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), 1, cv2.LINE_AA)

                                # if final_direction == -1:
                                #     text = "Walk right"
                                #     cv2.putText(overlay, text, (45, 350), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                #     1, (0, 0, 255), 1, cv2.LINE_AA)
                                #     cv2.rectangle(overlay, (0, 0), (300, 400), (0, 0, 255), 3)
                                #     #cli.send_alert("stay right")
                                #     print(text)

                                # elif final_direction == 1:
                                #     text = "Walk left"
                                #     cv2.putText(overlay, text, (45, 350), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                #     1, (0, 0, 255), 1, cv2.LINE_AA)
                                #     cv2.rectangle(overlay, (0, 0), (300, 400), (0, 0, 255), 3)
                                #     #cli.send_alert("stay left")
                                #     print(text)
                                # elif deviate:
                                #     text = "Go ahead"
                                #     cv2.putText(overlay, text, (45, 350), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                #     1, (0, 0, 255), 1, cv2.LINE_AA)
                                #     cv2.rectangle(overlay, (0, 0), (300, 400), (0, 0, 255), 3)
                                # else:
                                #     text = "Go ahead"
                                #     cv2.putText(overlay, text, (45, 350), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                #     1, (0, 0, 255), 1, cv2.LINE_AA)

                                #障礙物警告文字
                                if obstacle_check:
                                    text = "Obstacle !!"
                                    cv2.putText(overlay, text, (45, 380), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), 1, cv2.LINE_AA)
                                #紅色外框警示
                                if deviate or obstacle_check:
                                    cv2.rectangle(overlay, (0, 0), (330, 400), (0, 0, 255), 3)
                                

                            im0 = cv2.addWeighted(overlay, 1, im0, 0, 0)
                            label = 'Target' if find_target else label
                            # label = label + ', Degree:' + str(orientation)
                            # label = label + ' '+ corrected_dict[corrected_direction] if deviate else label
                            # label = label + ' Obstacle detected' if obstacle_check else label

                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    # # 根據用戶的位置更新Block的顏色(紅色區塊)
                    overlay = im0.copy()
                    cv2.drawContours(overlay, [pts_blocks[block_id]], -1, (0, 0, 255), thickness=-1)
                    im0 = cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0)
                    # 換下一秒的rssi
                    if not fps_count%30:
                        data_id += 1 
                
                fps_count += 1
                # Print time (inference + NMS)
                # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    #cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            #print(f'width : {w}, height : {h}')
                        vid_writer.write(im0)
           
            # 判斷用戶是否已經離開，若counter為3表示已離開，清除紀錄內容`
            if last_xy:
                for key, record in last_xy.items():
                    if record[5] == 3:
                        pass
                        #del last_xy[key]
                    else:
                        record[5] += 1
                        last_xy[key] = record
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")
    
    # bias_df.to_csv('0722test2_5.csv', mode='a', date_format='%H:%M:%S', index=False)
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
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
    #BLE
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    # optional
    parser.add_argument('--beforeDir',
                        help='File path of the csv before preprocessing',
                        type=str,
                        default='')

    parser.add_argument('--afterDir',
                        help='File path of the csv after preprocessing',
                        type=str,
                        default='')

    parser.add_argument('--timestep',
                        help='set time step',
                        type=int,
                        default=0)

    opt = parser.parse_args()
    #print(opt)
    update_config(cfg, opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect(cfg)
