import os 
import numpy as np
import cv2
import torch
import sys
 
def transfer(img_file):
	img = cv2.imread(img_file)[120:1000-120, 120:1000-120, :]
	img = cv2.resize(img, (128,128))
	img = np.transpose(img, (2, 0, 1))
	img = torch.from_numpy(img).unsqueeze(dim=0)
	return img
	

data_path = r'G:\datasets\original\Human3.6M\S8'
save_root = r'G:\datasets\original\Human3.6M\Pretreatment\S8'
files = os.listdir(data_path)
files.sort()
print(files[0])

index = 0
temp = []
prev_seq = '54138969' # Select the first sequence number of each dataset

for file in files:
	cur_path = os.path.join(data_path, file)

	#dirs = file.split('.')[0].split('_')
	action = file.split('.')[0]
	save_dir = os.path.join(save_root, action)

		
	seq = file.split('.')[1].split('_')[0]

	
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	if seq != prev_seq and temp:
		print(index)
		data = torch.cat(temp, dim=0)
		torch.save(data, os.path.join(save_dir, prev_seq + '.pth'))
		temp.clear()
		prev_seq = seq
		index += 1
	temp.append(transfer(cur_path))

