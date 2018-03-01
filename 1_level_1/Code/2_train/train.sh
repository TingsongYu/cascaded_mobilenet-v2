#!/usr/bin/env sh
set -e
postfix=`date +"%F-%H-%M-%S"`
/***your_caffe_path***/build/tools/caffe train \
--solver=./solver.prototxt -gpu 0,1  \
2>&1 | tee ../../Result/log/$(date +%Y-%m-%d-%H-%M.log) $@