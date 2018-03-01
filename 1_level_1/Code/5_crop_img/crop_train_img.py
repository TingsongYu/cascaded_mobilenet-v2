# -*- coding: utf-8 -*-
import sys
sys.path.append('../../../util')
import tools
import numpy as np
import argparse
import glob
import matplotlib.pyplot as plt
import os
import cv2

l1_out_label = '../../Result/l1_out_train_label.txt'
l1_raw_label = '../../Data/l1_train_label.txt'
relative_path = '../../../raw_data/Data/img_celeba/' # find the image from txt

crop_img_flod = '../../../level_2/Data/l1_crop/train/'

crop_label_flod = '../../../level_2/Data/l1_crop/' 
crop_label_txt =  crop_label_flod + 'l1_crop_train_label.txt'
crop_draw_img_flod = '../../../level_2/Data/l1_crop_draw/train/'
tools.makedir(crop_img_flod)

n_p = 5
# ----------------------------------------------------------------------- load label
l1_raw_fid = open(l1_raw_label)		
l1_raw_lines = l1_raw_fid.readlines()
l1_raw_fid.close()
l1_out_fid = open(l1_out_label)
l1_out_lines = l1_out_fid.readlines()
l1_out_fid.close()
err_mat = []

threshold = 0.1
count_threshold = 0
fid = open(crop_label_txt,'w')
for idx in range(len(l1_out_lines)):
	print idx
	r_ = l1_raw_lines[idx]			
	o_ = l1_out_lines[idx]
	r_name = r_.split()[0]
	o_name = o_.split()[0]
	if r_name != o_name: 
		print 'find a error,idx: ', idx 
		continue
	full_img_path = relative_path + r_name
	img = cv2.imread(full_img_path)
	h,w,c = img.shape	
	# ---------------------------------------------------------------------- calculate error
	err_1,err_5 = tools.cal_error_nor_diag(img,r_,o_)	# r_ have img name , range of [-1,1]  err_1 is mean 
	err_mat.append(err_5)

	raw_land = np.array(map(float,r_.split()[1:2*n_p+1]))	# nparray float
	out_land = np.array(map(float,o_.split()[1:2*n_p+1]))

	if err_1 < threshold :	
		# ------------------------------------------------------------  calculate w,h for crop img
		raw_pix_land = tools.label2points(raw_land,w,h)
		out_pix_land = tools.label2points(out_land,w,h)

		p_nose = out_pix_land[4:6]
		p_lefteye = out_pix_land[0:2]
		d_nose_lefteye = tools.cal_eucldist(p_nose,p_lefteye)

		w_start = np.round(p_nose[0] - 2*d_nose_lefteye).astype(int)
		w_end =   np.round(p_nose[0] + 2*d_nose_lefteye).astype(int)
		h_start = np.round(p_nose[1] - 2*d_nose_lefteye).astype(int)
		h_end =   np.round(p_nose[1] + 2*d_nose_lefteye).astype(int)

		if w_start < 0: w_start = 0
		if h_start < 0: h_start = 0
		if w_end > w: w_end = w
		if h_end > h: h_end = h
		# ------------------------------------------------------------  calculate new label
		crop_pix_land = raw_pix_land.copy()
		crop_pix_land[0::2] = crop_pix_land[0::2] - w_start	# x
		crop_pix_land[1::2] = crop_pix_land[1::2] - h_start # y

		crop_w = w_end - w_start
		crop_h = h_end - h_start
		w1 = (crop_w-1)/2
		h1 = (crop_h-1)/2
		crop_land = crop_pix_land.copy()
		crop_land[0::2] = (crop_pix_land[0::2] - w1) / w1
		crop_land[1::2] = (crop_pix_land[1::2] - h1) / h1
		# ----------------------------------------------------------- output crop img 
		crop_img = img.copy()
		crop_img = crop_img[h_start:h_end+1,w_start:w_end+1,:]
		crop_img_name = r_name
		crop_img_path = crop_img_flod + crop_img_name
		tools.makedir(crop_img_flod)
		cv2.imwrite(crop_img_path,crop_img)
		# ----------------------------------------------------------- output crop draw img 
		crop_draw_img = crop_img.copy()
		crop_draw_img = tools.drawpoints(crop_draw_img, crop_land)	
		crop_draw_img_name = r_name
		crop_draw_img_path = crop_draw_img_flod + crop_draw_img_name
		tools.makedir(crop_draw_img_flod)
		cv2.imwrite(crop_draw_img_path,crop_draw_img)
		# ----------------------------------------------------------- output label
		new_line = r_name 
		str_0 = str(crop_land)
		str_1 = str_0.replace("\n","")
		str_2 = str_1.strip('[]')
		str_3 = str_2.split()
		for i in range(n_p):
			x_ = str_3[2*i+0] # value is [-1,1] 
			y_ = str_3[2*i+1]

			new_line = new_line + ' ' + str(x_)			# note: the point order has changed: x1,y1,x2...
			new_line = new_line + ' ' + str(y_) 
		new_line = new_line +  '\n'
		fid.write(new_line)
fid.close()