#!/usr/bin/env bash

DATADIR='data' #location where data gets downloaded to

# get data
mkdir -p $DATADIR && cd $DATADIR
wget --content-disposition https://cg.cs.tsinghua.edu.cn/dataset/subdivnet/datasets/Manifold40.zip
unzip -q Manifold40.zip && rm Manifold40.zip
echo "downloaded the data and putting it in: " $DATADIR
