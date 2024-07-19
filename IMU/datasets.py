import pandas as pd
import torch.utils.data as data
import time

class IMU_Dataset(data.Dataset):

    def __init__(self, file) -> None:

        df = pd.read_csv(file)
        selected_columns = ['Timestamp', 'Yaw']
        df = df[selected_columns]
        df = df.groupby(df.columns[0]).mean().reset_index()
        df.to_csv('result.csv', index=False)

        self.time = df['Timestamp']
        self.yaw = df['Yaw']
        self.total_length = len(self.yaw)

    def __getitem__(self, index):
        time = self.time[index]
        yaw = self.yaw[index]
        return time, yaw

    def __len__(self):
        return self.total_length


