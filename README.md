# cascaded_mobilenet-v2
cascaded convolutional neural network for facial point detection

# 1.简介
本实验在caffe下，采用级联MobileNet-V2进行人脸关键点（5点）检测，单模型仅 956 KB，GTX1080上运行为6ms左右（未进行模型压缩和加速，简单压缩和加速后可在移动端达到实时检测）

本实验采用两级MobileNet-V2进行，两级的MobileNet-V2采用相同的网络结构（因为懒），结构如下：

| Input     |    Operator    | t  |c      |    n | s  |
| :--------:| :--------:| :--: |:--------:| :--------:| :--: |
| 48x48x3  | conv2d |  -   | 16  | 1 |  2   |
| 24x24x16  | bottleneck |  6   | 24  | 2 |  2   |
| 12x12x24  | bottleneck |  6   | 32  | 2 |  2   |
| 6x6x32 | bottleneck |  6   | 64  | 2 |  2   |
| 3x3x64  | fc |  -   | 256  | - |  -   |
| 1x1x256  | fc |  -   | 10  | - |  -   |

（MobileNet-v2 原文： https://arxiv.org/abs/1801.04381）

基本流程为，level_1负责初步检测，依据level_1得到的关键点，对原始图片进行裁剪，将裁剪后的图片输入到level_2，从而达到从粗到精的定位。
## level_1 流程为：
![image](https://github.com/tensor-yu/cascaded_mobilenet-v2/blob/master/readme_img/l1.PNG)

## level_2 流程为
![image](https://github.com/tensor-yu/cascaded_mobilenet-v2/blob/master/readme_img/l2.PNG)

 面部放大，绿色点为landmark，红色为level_1检测到的点，蓝色为level_2检测到的点，可以看出蓝色点更靠近绿色点

![image](https://github.com/tensor-yu/cascaded_mobilenet-v2/blob/master/readme_img/ccnntexie.PNG)



本实验初步验证MobileNet-V2的有效性以及级联CNN进行人脸关键点检测的有效性

数据来源：采用CelebA数据集，共计202599张图片，每张图片含5个关键点
官网：http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
百度网盘下载：https://pan.baidu.com/s/1eSNpdRG#list/path=%2F

实验结果：请直接看demo跑出来的图片。由于CelebA的图片较为复杂，并且本实验不需要采用人脸检测，因此无法与之前实验进行比较

# 2.运行demo

## (1) 修改 caffe源码
本实验基于MobileNet-V2，因此需要给caffe添加新的layer，即depth-wise convolution，并且需要修改image_data_layer，使得其支持多标签输入
(感谢 hpp,cpp,cu，prototxt提供者：suzhenghang  git地址：https://github.com/suzhenghang/MobileNetv2/tree/master/.gitignore)

步骤,进入caffe_need/文件夹下，

1. 将image_data_layer.hpp 替换掉 ***caffe_path***/include/caffe/layers 下的 image_data_layer.hpp
2. 将conv_dw_layer.hpp    复制到 ***caffe_path***/include/caffe/layers 下
3. 将image_data_layer.cpp 替换掉 ***caffe_path***/src/caffe/layers 下的image_data_layer.cpp
4. 将conv_dw_layer.cu
	 conv_dw_layer.cpp    复制到 ***caffe_path***/src/caffe/layers 下
重新编译，并且配置python接口


## (2) 直接进入 3_demo
3_demo文件夹下含 Code、Data. Data中有deploy.prototxt、caffemodel和img
进入 3_demo/Code/,打开 inference , 更改你的caffe所在路径

	sys.path.append('/home/xxx your caffe xxx/python')
	sys.path.append('/home/xxx your caffe xxx/python/caffe')

然后运行  sudo python inference.py, 检测出的图片保存在 3_demo/Result/draw_img/ 下

# 3.复现训练过程
简单介绍训练步骤，总共分三阶段，分别是 0_raw_data, 1_level_1, 2_level_2 

第一阶段，数据准备阶段： 0_raw_data 
1. 从百度网盘下载好CelebA数据集，将CelebA\Img\img_celeba 复制到 0_raw_data/Data/ 下面，将CelebA\Anno\list_landmarks_celeba.txt复制到  0_raw_data/Data/ 并且重命名为celeba_label.txt
2. 进入0_raw_data/, 运行divide_tr_te.py，将会划分好训练集，测试集，并且保存在0_raw_data/Data/ 下面 
3. 运行 draw_point.py，将会在 0_raw_data/Result/draw_img/下获得 打上关键点的图片，用来检查图片以及标签是否正确


第二阶段, 训练level_1： 1_level_1 

	进入 1_level_1/Code/，依次执行 0_gen_data, 1_draw_img, 2_train, 3_inference, 4_evaluate, 5_crop_img
	0_gen_data，主要是对图片进行resize，并且转换label，训练时的label是[-1,1]的
	1_draw_img,用来检查图片以及标签是否正确
	2_train,训练的solver等 
	3_inference,训练完毕，用训练好的caffemodel进行inference，将inference得到的标签 输出到 1_level_1/Result/下，用于评估和裁剪图片
	4_evaluate，计算误差
	5_crop_img, 采用level_1的输出标签 对原始图片进行裁剪，获得level_2的输入图片，并且制作level_2的标签


第三阶段，训练level_2，
由于 1_level_1/Code/5_crop_img 已经生成了 level_2所需的数据，并且打上关键点，供检查，因此 level_2直接从train开始

0_train, 同level_1
1_inference, 同level_1
2_evaluate,同level_1








