import os 
import numpy as np
import cv2
import torch
import sys
 
def transfer(imgs_path):
	img_files = os.listdir(imgs_path)
	img_files.sort()
	print(imgs_path)
	left = []
	mid = []
	right = []
	for img_file in img_files:
		img = cv2.imread(os.path.join(imgs_path, img_file))
		img_mid = img[:, 184:184+1024] # center crop
		img_mid = cv2.resize(img_mid, (256,128))
		img_mid = np.transpose(img_mid, (2, 0, 1))
		img_mid = torch.from_numpy(img_mid).unsqueeze(dim=0)
		mid.append(img_mid)

	data_mid = torch.cat(mid, dim=0)

	return data_mid
	

data_path = r'G:\datasets\original\KITTI\image\road'
save_root = r'G:\datasets\original\KITTI\Pretreat2'
files = os.listdir(data_path)
files.sort()
index = 0
temp = []


for i in range(len(files)):
	cur_path = os.path.join(data_path, files[i], 'data')
	
	save_dir = os.path.join(save_root, data_path.split('\\')[-1])
	
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
		
	mid = transfer(cur_path)
	torch.save(mid, os.path.join(save_dir, '{}_mid.pth'.format(i)))

