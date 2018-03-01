# divide celebA dataset
import sys
sys.path.append('../../util')
import tools
import os
import random
import shutil

raw_txt = '../Data/celeba_label.txt'
relative_path =  '../Data/img_celeba/'  #   for  find the img 
train_txt = '../Result/raw_train_label.txt'		# target txt
test_txt = '../Result/raw_test_label.txt'
train_img_fold = '../Result/train/'		
test_img_fold = '../Result/test/'
tools.makedir(train_img_fold)
tools.makedir(test_img_fold)

per = 0.8 	# percentage of train set 
line_num = 0
train_num = 0
test_num = 0
train_f = open(train_txt,"w")
test_f = open(test_txt,"w")
for line in open(raw_txt):	
	if line.isspace() : continue  # skip empty line 
	line_num += 1
	img_name = line.split()[0]
	full_img_path = relative_path + img_name
	a_rand = random.uniform(0,1)	
	# train set
	if a_rand <= per:			
		train_f.write(line)
		train_img_path = train_img_fold + img_name
		shutil.copy(full_img_path,train_img_path)
		train_num += 1
	# test set
	else:
		test_f.write(line)
		test_img_path = test_img_fold + img_name
		shutil.copy(full_img_path,test_img_path)
		test_num +=1
	print 'img : ', line_num
train_f.close()
test_f.close()



print 'train set have ', train_num ,' examples.'
print 'test  set have ', test_num , ' examples.' 
print  train_num ,' + ' ,test_num ,' = ', train_num+test_num
print 'line_num is ', line_num