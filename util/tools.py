# -*- coding: utf-8 -*-
import sys
import numpy as np
import os
import cv2

# range of [-1,1]  to  [0,w] or [0,h]
def convert_point(nor_p,factor):
	return int(round( float(nor_p)*factor + factor))

# draw points for level_1
# landmark range of [-1,1] 		x1,y1,x2,y2...
def drawpoints_1(img_,landmark):
	h, w, c = img_.shape
	w1 = (w-1)/2	# range of [-1, 1]
	h1 = (h-1)/2	
	draw_img = img_.copy()
	num_points  =  len(landmark) / 2 
	for i in range(num_points):
		x_ = convert_point(landmark[2*i+0],w1)
		y_ = convert_point(landmark[2*i+1],h1)
		cv2.circle(draw_img,(x_,y_),3,(0,0,255))
	return draw_img

# draw points for level_2
# landmark range of [0,h] or [0,w] 	x1,y1,x2,y2...
def drawpoints_2(img_,landmark): 
	draw_img = img_.copy()
	num_points  =  len(landmark) / 2 
	for i in range(num_points):
		x_ = landmark[2*i+0]
		y_ = landmark[2*i+1]
		cv2.circle(draw_img,(x_,y_),3,(255,0,0))
	return draw_img

# draw points for level_2
# landmark range of [0,h] or [0,w] 	x1,y1,x2,y2...
def drawpoints_0(img_,landmark):
	draw_img = img_.copy()
	num_points  =  len(landmark) / 2 
	for i in range(num_points):
		x_ = myint(landmark[2*i+0])
		y_ = myint(landmark[2*i+1])
		cv2.circle(draw_img,(x_,y_),4,(0,255,0)) # green 
	return draw_img

def myint(numb):
	return int(round(float(numb)))

def cal_eucldist(v1,v2):
	return np.sqrt(np.sum((v1-v2)**2))

def makedir(path):
	if not os.path.exists(path):  os.makedirs(path)

# label change to pixel
# l range of [-1,1] ; len(l) = 10
def label2points(l,w,h):
	landmark = l.copy()	 
	num_points = len(landmark) /2 
	w1 = (w-1)/2	# range of [-1, 1]
	h1 = (h-1)/2
	landmark[0::2] = landmark[0::2]*w1 +w1 	# x
	landmark[1::2] = landmark[1::2]*h1 +h1 	# y
	landmark = np.round(landmark)
	return landmark

def cal_dist_norm_eye(landmark):
	left_eye = landmark[0:2]
	right_eye = landmark[2:4]
	return cal_eucldist(left_eye, right_eye)

# 
# r_l range of [-1,1]  
# # err_1 is mean error 
def cal_error_nor_diag(img,r_l,o_l):
	h,w,c = img.shape
	n_p = 5
	r_landmark = np.array(map(float,r_l.split()[1:2*n_p+1]))
	o_landmark = np.array(map(float,o_l.split()[1:2*n_p+1]))
	r_pix_landmark = label2points(r_landmark,w,h)
	o_pix_landmark = label2points(o_landmark,w,h)	

	d_diag = np.sqrt(w*w + h*h)
	err_all = 0
	err_5 = [] 
	for i in range(n_p):
		raw_point = r_pix_landmark[2*i+0:2*i+2]
		out_point = o_pix_landmark[2*i+0:2*i+2]
		err_ecul = cal_eucldist(raw_point, out_point) / d_diag
		err_all = err_all + err_ecul
		err_5.append(err_ecul)
	err_1 = round(err_all / n_p ,4)	# mean 
	return err_1,err_5


#  crop_img for level_2 
def crop_img(in_img,in_land):
	p_nose = in_land[4:6]		
	p_lefteye = in_land[0:2]
	d_nose_lefteye = cal_eucldist(p_nose,p_lefteye)

	w_start = np.round(p_nose[0] - 2*d_nose_lefteye).astype(int)
	w_end =   np.round(p_nose[0] + 2*d_nose_lefteye).astype(int)
	h_start = np.round(p_nose[1] - 2*d_nose_lefteye).astype(int)
	h_end =   np.round(p_nose[1] + 2*d_nose_lefteye).astype(int)

	h_img,w_img,c = in_img.shape

	if w_start < 0: w_start = 0
	if h_start < 0: h_start = 0
	if w_end > w_img: w_end = w_img
	if h_end > h_img: h_end = h_img

	crop_img = in_img.copy()
	crop_img = crop_img[h_start:h_end+1,w_start:w_end+1,:]
	return crop_img,w_start,h_start

# for ALFW   x1 x2...   --->  x1 y1 x2 y2
def change_order(in_land):
	n_p = len(in_land)/2
	out_land = in_land[:]
	for i in range(n_p):
		out_land[2*i+0] = in_land[i]
		out_land[2*i+1] = in_land[i+5]
	return out_land




















