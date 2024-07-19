##執行指令 
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source videos.txt --cfg BLE\config\U19e_outdoor0805test2_3.yaml  --camera_num 1

--source 為串流來源 (streams.txt 為真實監視器串流; videos.txt 裡面放 video 的檔案路徑)
--camera_num 為監視器數量 (需跟 streams.txt 或 videos.txt的串流來源數量一致)


需另外下載 re id 和 MEBOE 資料夾並解壓縮

re id 連結:
https://drive.google.com/file/d/12O8SRJhwJBKCIbIDegetJTa-X3B0Ejmn/view?usp=drive_link

MEBOE 連結:
https://drive.google.com/file/d/1sSKSFBMdofwuTNjH3ZFsITEYs_AJU_H7/view?usp=drive_link

