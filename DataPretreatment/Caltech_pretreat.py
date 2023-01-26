
''' Dataset can be aquired from: https://drive.google.com/drive/folders/1cnQHqa8BkVx90-6-UojHnbMB0WhksSRc'''

# -*- coding:utf-8 -*-
import os
import fnmatch
import sys
import cv2
import torch
import numpy as np

def open_save(file, savepath):
    f = open(file, 'rb')
    string = f.read().decode('latin-1')
    splitstring = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"  
    strlist = string.split(splitstring)
    f.close()
    count = 0  

    for img in strlist:        
        filename = str(count)+'.jpg'
        filename = os.path.join(savepath, filename)
        # print(savepath,filename)
        # sys.exit()
        if count > 0:                                    
            i = open(filename, 'wb+')
            i.write(splitstring.encode('latin-1'))
            i.write(img.encode('latin-1'))
            i.close()
        count = count + 1
        # print("Generating JPEGImages jpg file of picture:{}".format(filename))


def main():
	jpg_outputdir = r"E:\datasets\original\Caltech\image"
	seq_inputdir = r"E:\datasets\original\Caltech\seq"
	for cur_dir, dirs, files in os.walk(seq_inputdir):
		if len(files) == 0:
			continue
		save_dir = os.path.join(jpg_outputdir, cur_dir.split('\\')[-1])
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)
		for file in files:
			print(cur_dir.split('\\')[-1], file)
			save_path = os.path.join(save_dir, file.split('.')[0])
			if not os.path.exists(save_path):
				os.mkdir(save_path)
			file_path = os.path.join(cur_dir, file)
			open_save(file_path, save_path)
			
def transfer(img_file):
	img = cv2.imread(img_file)
	img = cv2.resize(img, (160,128))
	#cv2.imshow('img',img)
	#cv2.waitKey(5)
	img = np.transpose(img, (2, 0, 1))
	img = torch.from_numpy(img).unsqueeze(dim=0)
	return img
	

def ToTensor():
	img_dir = r'E:\datasets\original\Caltech\image'
	save_root = r'E:\datasets\original\Caltech\Pretreatment'
	for cur_dir, dirs, files in os.walk(img_dir):
		if len(files) == 0:
			continue
		set_index, seq = cur_dir.split('\\')[-2:]
		print(set_index, seq)
		set_path = os.path.join(save_root, set_index)
		if not os.path.exists(set_path):
			os.mkdir(set_path)
		save_path = os.path.join(set_path, seq + '.pth')
		temp = []
		files.sort(key=lambda x:int(x.split('.')[0]))
		for file in files:
			img_file = os.path.join(cur_dir, file)
			img = transfer(img_file)
			temp.append(img)
		data = torch.cat(temp, dim=0)
		torch.save(data, save_path)
		
			 

if __name__ == '__main__':
    #main()
    #print("Success！")
    ToTensor()
    
