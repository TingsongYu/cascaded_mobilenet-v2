# generate img and txt for level_1
# The point order  x1,x2,x3...
import sys
sys.path.append('../../util')
import tools
import os
import numpy as np
import cv2

train_txt = '../Result/raw_train_label.txt'  #  raw_txt
test_txt = '../Result/raw_test_label.txt'   
relative_path = '../Data/img_celeba/'
draw_dir = '../Result/draw_img/' #  
tools.makedir(draw_dir)

n_p = 5 # num of points

def myint(numb):
	return int(round(float(numb)))
def drawpoint(raw_txt,o_dir):
	for line in open(raw_txt):
		if line.isspace() : continue  # 
		raw_land = list(line.split())[1:2*n_p+1]

		img_name = line.split()[0] 
		full_img_path = relative_path + img_name		
		img = cv2.imread(full_img_path)
		draw_img = img.copy()
		draw_img = tools.drawpoints_0(draw_img,raw_land)

		# output img
		sub_flod = o_dir + raw_txt.split('_')[-2]		
		tools.makedir(sub_flod)
		draw_img_path = sub_flod + '/' + img_name
		print (draw_img_path)
		cv2.imwrite(draw_img_path,draw_img)
	open(raw_txt).close()
	print(raw_txt,' done!')

drawpoint(test_txt,draw_dir)
drawpoint(train_txt,draw_dir)