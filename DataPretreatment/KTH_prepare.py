
'''
Dataset available:
Original homepage:  https://www.csc.kth.se/cvap/actions/
or get it from here:
https://drive.google.com/drive/folders/12sK0ZTscHOAXnSGHkFBmFBM9XawpWYlK?usp=sharing
'''
import os 
import numpy as np
import cv2
import torch
import sys
 
def splitFrames(videoFileName):
    cap = cv2.VideoCapture(videoFileName) # 打开视频文件
    num = 1
    temp = []
    while True:
        # success 表示是否成功，data是当前帧的图像数据；.read读取一帧图像，移动到下一帧
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (128,96))
        frame = np.transpose(frame, (2,0,1))
        frame = torch.from_numpy(frame).unsqueeze(dim=0)
        temp.append(frame)      
    cap.release()
    data = torch.cat(temp, dim=0)
    temp.clear()
    return data
 

data_path = r'E:\datasets\original\KTH\KTH_raw\KTH_avi'
save_root = r'E:\datasets\original\KTH\KTH_raw\PreTreatment'
video_files = os.listdir(data_path)
index = 0
for file in video_files:
	print(index)
	cur_path = os.path.join(data_path, file)
	save_dir = os.path.join(save_root, file.split('_')[0])
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	name = list(file.split('_')[1:3])
	name.insert(1,'_')
	name.append('.pth')
	save_name = "".join(name)
	data = splitFrames(cur_path)
	torch.save(data, os.path.join(save_dir, save_name))
	index += 1
