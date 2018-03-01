# generate img and txt for level_1
# The point order has changed: x1,y1,x2...
import sys
sys.path.append('../../../util')
import tools
import os
import numpy as np
import cv2
train_txt = '../../../raw_data/Result/raw_train_label.txt' 		#  raw_txt
test_txt = '../../../raw_data/Result/raw_test_label.txt'     
l1_data_dir = '../../Data/' 									# target dir
l1_train_txt = l1_data_dir + 'l1_train_label.txt'				# target txt
l1_test_txt = l1_data_dir + 'l1_test_label.txt'
relative_path =  '../../../raw_data/Data/img_celeba/'  			#  for  find the img 
tools.makedir(l1_data_dir)

net_1_w = 48
net_1_h = 48
n_p = 5 # num of points
def gendata(target_txt,raw_txt):
	with open(target_txt,"w") as f:
		for line in open(raw_txt):
			# txt 
			if line.isspace() : continue
			img_name = line.split()[0] 
			full_img_path = relative_path + img_name
			print full_img_path
			img = cv2.imread(full_img_path)

			w = img.shape[1]		# weight is x axis
			h = img.shape[0]		# height is y axis
			w1 = (w-1)/2			# for  [-1,1]
			h1 = (h-1)/2 

			raw_land = list(line.split())[1:2*n_p+1] 
			new_line = img_name
			for i in range(n_p):
				x_ = round( (float(raw_land[2*i+0])-w1)/w1 , 4)  # value is [-1,1] 
				y_ = round( (float(raw_land[2*i+1])-h1)/h1 , 4)	
				new_line = new_line + ' ' + str(x_)			# note: The point order has changed: x1,y1,x2...
				new_line = new_line + ' ' + str(y_) 
			print('new_line: ', new_line)
			f.write(new_line + '\n') 

			# image 
			scale_img = cv2.resize(img,(net_1_w,net_1_h))
			sub_flod = l1_data_dir + raw_txt.split('_')[2] + '/'	
			tools.makedir(sub_flod)
			scale_img_path =  sub_flod + img_name
			print 'output path ',scale_img_path
			cv2.imwrite(scale_img_path,scale_img)
			# print a 
		open(raw_txt).close()
gendata(l1_test_txt,test_txt)
gendata(l1_train_txt,train_txt)

