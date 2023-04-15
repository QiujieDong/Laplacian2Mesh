#!/usr/bin/env bash

DATADIR='data' #location where data gets downloaded to

# get data
mkdir -p $DATADIR && cd $DATADIR
wget --content-disposition https://cg.cs.tsinghua.edu.cn/dataset/subdivnet/datasets/Manifold40-MAPS-96-3.zip
unzip -q Manifold40-MAPS-96-3.zip && rm Manifold40-MAPS-96-3.zip
echo "downloaded the data and putting it in: " $DATADIR
