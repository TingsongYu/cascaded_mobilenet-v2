# -*- coding: utf-8 -*-
import sys
sys.path.append('../../../util')
sys.path.append('/***your_caffe_path***/python')
sys.path.append('/***your_caffe_path***/python/caffe')
import tools
import caffe
import numpy as np
import cv2
import time

l1_deploy = './l1_deploy.prototxt'
l1_model = '../../Result/solver_state/_iter_100000.caffemodel'

txt_flod = '../../Data/'
train_txt = txt_flod + 'l1_train_label.txt'
test_txt = txt_flod + 'l1_test_label.txt'

relative_path = '../../../raw_data/Data/img_celeba/' 	# find the image

l1_out_train_txt = '../../Result/l1_out_train_label.txt'
l1_out_test_txt = '../../Result/l1_out_test_label.txt'

w_net = 48
h_net = 48
#--------------------------------------------------------------------------- cnn initalization
caffe.set_mode_gpu()
caffe.set_device(0)
# load model 
net = caffe.Net(l1_deploy,l1_model,caffe.TEST)
# image preprocess
mu = np.ones((3,w_net,h_net), dtype=np.float) * 127.5
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) 
transformer.set_transpose('data', (2,0,1))  # (w,h,c)--> (c,w,h)
transformer.set_mean('data', mu)            # pixel-wise 
transformer.set_raw_scale('data', 255 )      # [0,1] --> [0,255]
transformer.set_channel_swap('data', (2,1,0)) # RGB -->  BGR
#----------------------------------------------------------------------------- forward
def l1_forward(input_txt,output_txt):
	out_f = open(output_txt,'w')
	for line in open(input_txt):
		if line.isspace() : continue  
		img_name = line.split()[0] 
		full_img_path = relative_path + img_name		
		#------------------------------------------------------------------------- cnn forward 
		im=caffe.io.load_image(full_img_path)  # im is  RGB  with  0~1 float
		net.blobs['data'].data[...]=transformer.preprocess('data',im)
		time_s = time.clock()
		n_out = net.forward()
		time_e = time.clock()
		print img_name,'forward : ',round((time_e-time_s)*1000,1) ,'ms'
		out_landmark = net.blobs['fc2'].data[0].flatten()
		#------------------------------------------------------------------------- write txt
		str_0  = str(out_landmark)
		str_1 = str_0.replace("\n","")
		str_2 = str_1.strip('[]')
		new_line = img_name +' '+ str_2 +'\n'
		out_f.write(new_line)
	out_f.close()

l1_forward(test_txt,l1_out_test_txt)
l1_forward(train_txt,l1_out_train_txt)