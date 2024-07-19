import requests
import time

# 设置服务器的地址
server_url = "http://140.116.72.77:5000/imu"

# 准备要发送的数据
data_to_send = {'imu': 72}

t0 = time.time()

# 发送 POST 请求
response = requests.post(server_url, data=data_to_send)

t1 = time.time()

print(f"Time: {t1 - t0}")

# 解析响应
if response.status_code == 200:
    # 请求成功
    result = response.json()
    print(result)
else:
    # 请求失败
    print(f"Error: {response.status_code}, {response.text}")
