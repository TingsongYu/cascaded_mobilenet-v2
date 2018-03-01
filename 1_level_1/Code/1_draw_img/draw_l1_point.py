# draw points for level_1 
import sys
sys.path.append('../../../util')
import tools
import os
import numpy as np
import cv2

relative_path =  '../../Data/'  					# for  find the img 
relative_train_path = '../../Data/train/'
relative_test_path = '../../Data/test/'

train_txt = relative_path + 'l1_train_label.txt'  	#  raw_txt
test_txt = relative_path + 'l1_test_label.txt' 

draw_dir = relative_path + 'draw_img/'
tools.makedir(draw_dir)
n_p = 5 # num of points

def drawpoint(raw_txt,o_dir,relative_img_path):
	for line in open(raw_txt):
		if line.isspace() : continue  # 
		img_name = line.split()[0] 
		full_img_path = relative_img_path + img_name
		img = cv2.imread(full_img_path)
		draw_img = img.copy()

		w = img.shape[1]		# width is x axis
		h = img.shape[0]		# height is y axis
		w1 = (w-1)/2			# for  [-1,1]
		h1 = (h-1)/2 	

		raw_land = list(line.split())[1:2*n_p+1]
		for i in range(n_p): 			# draw key points
			x_ = tools.convert_point(raw_land[2*i+0],w1)
			y_ = tools.convert_point(raw_land[2*i+1],h1)
			cv2.circle(draw_img,(x_,y_),2,(0,255,0))
		# output img
		sub_flod = o_dir + raw_txt.split('_')[-2] + '/'
		tools.makedir(sub_flod)			
		draw_img_path = sub_flod + img_name
		print 'draw ima path ', draw_img_path
		cv2.imwrite(draw_img_path,draw_img)
	open(raw_txt).close()
	print(raw_txt,' done!')
drawpoint(train_txt,draw_dir,relative_train_path)
drawpoint(test_txt,draw_dir,relative_test_path)
