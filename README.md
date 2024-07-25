# **執行指令**

python detect.py --weights yolov7.pt --conf 0.25 --img-size 640
--source videos.txt --cfg BLE\config\U19e_outdoor0805test2_3.yaml  --camera_num 1

--source 為串流來源 (videos.txt 裡面放 video 的檔案路徑)
--camera_num 為監視器數量 (需跟videos.txt的串流來源數量一致)

需另外下載 videos, re id 和 MEBOE 資料夾並解壓縮

videos 連結:
https://drive.google.com/file/d/1f4Aba-iVgqYi7bTFUJ4vUTlfINjIHc_b/view?usp=sharing

re id 連結:
https://drive.google.com/file/d/12O8SRJhwJBKCIbIDegetJTa-X3B0Ejmn/view?usp=drive_link

MEBOE 連結:
https://drive.google.com/file/d/1sSKSFBMdofwuTNjH3ZFsITEYs_AJU_H7/view?usp=drive_link

# **Highlevel overview of source files**

以下提供幾個較重要的程式功能和檔案解說

`detect.py` : 主程式，負責 Yolov7 偵測 重識別 持續追蹤 和 導盲等功能。<br>
`videos.txt` : 裡面放測試影片的檔案路徑。<br>
`videos` : 裡面放測試影片。<br>
`handover_paramter.py` : handover 相關，儲存畫面中的 hanover 區域範圍，判斷使用者是否在外圍中，以及判斷使用者的 IMU yaw 值有沒有在接受範圍。<br>
`tactile_parameter.py` : 儲存畫面中的導盲磚和安全區域範圍。<br>
`perspective_transformation.py` : perspective transformation 相關，儲存 perspective transformation 的轉換矩陣係數，以及轉換座標或安全範圍到另一個 2D 像素坐標系。<br>
`dashboard.py` : 畫障礙物相對位置和偏移矯正的向量到畫面上。<br>

`IMU\IMU_DATA` : 儲存 IMU 原始資料。<br>
`IMU\datasets.py` : 將 IMU 原始資料取出時間和 yaw 值並整理成一秒一筆。<br>
`result.csv` : 整理後的 IMU 資料。<br>

`re_id\model` : re id 模型的權重檔案。<br>
`re_id\model.py` : re id 模型。<br>
`re_id\Re_id_Matcher.py` : 重識別的配對器，根據使用者和行人的外觀特徵進行匹配。<br>

`Tracking\Re_idTracker.py` : 持續追蹤的配對器，根據使用者和行人的外觀特徵進行匹配(跟`re_id\Re_id_Matcher.py` 大同小異，主要差別在 Re_id_Matcher 在不同鏡頭做匹配，Re_idTracker 在同鏡頭的連續幀做匹配)<br>
`Tracking\KalmanTracker.py` : 持續追蹤的配對器，根據使用者和行人的 bounding boxes 用卡爾曼濾波器做匹配。<br>

`utils\datasets.py` : 主要用到裡面的 `LoadStreams` ， `LoadStreams` 會根據影片的 fps ，每過一段時間讀取每個影片並整理成影像給 `detect.py` 進行後續處理，另外也會根據影片的 fps 每過幾幀將 imu_dataset_id 加一 (也就是 `result.csv` 的下一秒資料)。(ex: fps 為 30 ， 每過 30 幀會過一秒，所以要將 imu_dataset_id 加一，讓 `detect.py` 可以讀到 `result.csv` 的下一秒資料)<br>