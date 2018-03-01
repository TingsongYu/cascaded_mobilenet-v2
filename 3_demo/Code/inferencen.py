# -*- coding: utf-8 -*-
import sys
sys.path.append('../../util')
sys.path.append('/***your_caffe_path***/python')
sys.path.append('/***your_caffe_path***/python/caffe')
import tools
import caffe
import numpy as np
import argparse
import cv2
import time

l1_deploy = '../Data/l1_deploy.prototxt'
l1_model = '../Data/l1_net.caffemodel'
l2_deploy = '../Data/l2_deploy.prototxt'
l2_model = '../Data/l2_net.caffemodel'
raw_txt = '../Data/demo.txt'
relative_path = '../Data/img/' 	# find the image
draw_img_flod = '../Result/draw_img/'
w_net = 48
h_net = 48
n_p = 5

#--------------------------------------------------------------------------- cnn initalization
# load model 
l1_net = caffe.Net(l1_deploy,l1_model,caffe.TEST)
l2_net = caffe.Net(l2_deploy,l2_model,caffe.TEST)

caffe.set_mode_gpu()
caffe.set_device(0)

# image preprocess
mu = np.ones((3,w_net,h_net), dtype=np.float) * 127.5
transformer = caffe.io.Transformer({'data': l1_net.blobs['data'].data.shape}) 
transformer.set_transpose('data', (2,0,1))  # (w,h,c)--> (c,w,h)
transformer.set_mean('data', mu)            # pixel-wise 
transformer.set_raw_scale('data', 255 )      # [0,1] --> [0,255]
transformer.set_channel_swap('data', (2,1,0)) # RGB -->  BGR
#----------------------------------------------------------------------------- forward
for line in open(raw_txt):
	if line.isspace() : continue  
	img_name = line.split()[0] 
	full_img_path = relative_path + img_name		
	img = cv2.imread(full_img_path)
	draw_img = img.copy()
	#-----------------------------------------------------------------------  l1 forward 
	l1_input_img=caffe.io.load_image(full_img_path)  # im is  RGB  with  0~1 float
	h_img,w_img,c = l1_input_img.shape

	l1_net.blobs['data'].data[...]=transformer.preprocess('data',l1_input_img)
	time_s = time.clock()
	l1_out = l1_net.forward()
	time_e = time.clock()
	print img_name,'l1_forward : ',round((time_e-time_s)*1000,1) ,'ms'
	l1_out_land = l1_net.blobs['fc2'].data[0].flatten()
	# crop img for level_2


	l1_out_pix_land = tools.label2points(l1_out_land,w_img,h_img)
	# ---------------------------------------------------------------------------- crop img
	crop_img,w_start,h_start = tools.crop_img(l1_input_img,l1_out_pix_land)
	#-----------------------------------------------------------------------  l2 forward 
	l2_input_img = crop_img
	h_l2,w_l2,c = l2_input_img.shape
	l2_net.blobs['data'].data[...]=transformer.preprocess('data',l2_input_img)
	time_s = time.clock()
	l2_out = l2_net.forward()
	time_e = time.clock()
	print img_name,'l2_forward : ',round((time_e-time_s)*1000,1) ,'ms'
	l2_out_land = l2_net.blobs['fc2'].data[0].flatten()
	l2_out_pix_land = tools.label2points(l2_out_land,w_l2,h_l2)

	l2_out_pix_land[0::2] = l2_out_pix_land[0::2] + w_start	# x
	l2_out_pix_land[1::2] = l2_out_pix_land[1::2] + h_start # y

	# --------------------------------------------------------------------     draw img 
	raw_land = list(line.split())[1:2*n_p+1]
	draw_img = tools.drawpoints_0(draw_img, raw_land)	
	draw_img = tools.drawpoints_1(draw_img, l1_out_land)	
	draw_img = tools.drawpoints_2(draw_img, l2_out_pix_land)

	# ---------------------------------------------------------------------    output img
	draw_img_path = draw_img_flod + img_name
	tools.makedir(draw_img_flod)
	cv2.imwrite(draw_img_path,draw_img)

