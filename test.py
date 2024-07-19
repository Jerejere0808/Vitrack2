import threading
import time
from IMU.datasets import increment_id

# 初始化變數
imu_dataset_id = {
    "id": 0,
    "lock": threading.Lock()
}

# 創建並啟動一個新 thread 來執行 increment_id 函數
thread = threading.Thread(target=increment_id, args=(imu_dataset_id,))
thread.daemon = True  # 將這個 thread 設為 daemon 使其在主程式結束時自動結束
thread.start()

# 主程式繼續運行，可以在這裡執行其他操作
try:
    while True:
        with imu_dataset_id["lock"]:
            print(f"Current imu_dataset_id: {imu_dataset_id['id']}")
        time.sleep(1)
except KeyboardInterrupt:
    print("程式結束")


