# -*- coding: utf-8 -*-
import sys
sys.path.append('../../../util')
import tools
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

l2_out_test_label = '../../Result/l2_out_test_label.txt'
l2_raw_test_label = '../../Data/l1_crop/l1_crop_test_label.txt'
relative_path = '../../Data/l1_crop/test/' 	# find the image from txt
draw_img_flod = '../../Result/l2_out_draw/test/'
drop_img_flod = '../../Result/l2_drop/test/'

n_p = 5
# ----------------------------------------------------------------------- load label
l2_raw_fid = open(l2_raw_test_label)		
l2_raw_lines = l2_raw_fid.readlines()
l2_raw_fid.close()
l2_out_fid = open(l2_out_test_label)
l2_out_lines = l2_out_fid.readlines()
l2_out_fid.close()

err_mat = []
threshold = 1
count_drop = 0
for idx in range(len(l2_out_lines)):
	print idx
	r_ = l2_raw_lines[idx]			
	o_ = l2_out_lines[idx]
	r_name = r_.split()[0]
	o_name = o_.split()[0]
	if r_name != o_name:
		print 'find a error,idx: ', idx 
		continue
	full_img_path = relative_path + r_name
	img = cv2.imread(full_img_path)
	h,w,c = img.shape	

	err_1,err_5 = tools.cal_error_nor_diag(img,r_,o_)	# r_ have img name , range of [-1,1]  err_1 is mean 
	err_mat.append(err_5)
	out_land = np.array(map(float,o_.split()[1:2*n_p+1]))
	
	if err_1 >= threshold :	
		count_drop = count_drop + 1
		draw_img = img.copy()
		draw_img = tools.drawpoints(draw_img,out_land)
		tools.makedir(drop_img_flod)
		draw_img_name = str(err_1) + '_' + r_name
		draw_img_path = drop_img_flod + draw_img_name
		cv2.imwrite(draw_img_path, draw_img)
	else:
		draw_img = img.copy()
		draw_img = tools.drawpoints(draw_img,out_land)
		tools.makedir(draw_img_flod)
		draw_img_name = str(err_1) + '_' + r_name
		draw_img_path = draw_img_flod + draw_img_name
		cv2.imwrite(draw_img_path, draw_img)
# -------------------------------------------------------------- print result
err_mat = np.array(err_mat)
err_mat = np.reshape(err_mat,(-1,5))
MNE_5 = []
for i in range(n_p):
	MNE_5.append(err_mat[:,i].mean())
print 'err >= 10%  have ' , count_drop
# ------------------------------------------------------------- plot 
fig = plt.figure('test_MNE_5')
ax1 =plt.subplot(111)
data  = np.array(MNE_5)
width = 0.2
x_bar = np.arange(5)
# print('x_bar type ',type(x_bar))
rect = ax1.bar(left=x_bar,height=data,width=width, color="blue")
for rec in rect:
	x= rec.get_x()
	height = round(rec.get_height()*100,2) 
	mne_text = str(height) + '%'
	# print('mne text',mne_text)
	ax1.text(x+0.05,1.02*height/100,mne_text)
	# print('height',height)
MNE_5_mean =  np.round(np.array(MNE_5).mean() *100,2)
MNE_5_mean_text = 'The mean normalized error :' +str(MNE_5_mean) + '%'
ax1.text(1 ,1.5*MNE_5_mean/100 ,MNE_5_mean_text,color="red")

ax1.set_xticks(x_bar + width)
ax1.set_xticklabels(("left eye","right eye","nose","left mouth","right mouth"))
ax1.set_ylabel("MNE")
ax1.set_title(" MNE")
ax1.grid(True)
ax1.set_ylim(0,0.025)		#  max y axis 
plt.show()



print 'The mean error normalized by dist_diag is : ', err_mat.mean()
# print  a
fig2 = plt.figure("test_distribution")
ax2 = plt.subplot(111)
ax2.set_title("The mean error normalized by dist_diag :")
data =err_mat.mean(axis=1)
n, bins, patches = plt.hist(data ,bins=200, normed=False, facecolor='blue', alpha=0.75)
err_mat_mean = np.round(np.array(err_mat).mean() *100 ,2)
mean_text = 'The mean error normalized by dist_diag : ' + str(err_mat_mean) + '%'
ax2.text(0.1,len(err_mat)/10 ,mean_text,color="red")
plt.show()