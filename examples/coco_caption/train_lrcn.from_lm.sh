#!/usr/bin/env bash

GPU_ID=2
WEIGHTS=\
../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel\
,\
snapshots/lstm_lm_iter_110000.caffemodel
DATA_DIR=../../examples/coco_caption/h5_data/
if [ ! -d $DATA_DIR ]; then
    echo "Data directory not found: $DATA_DIR"
    echo "First, download the COCO dataset (follow instructions in data/coco)"
    echo "Then, run ./examples/coco_caption/coco_to_hdf5_data.py to create the Caffe input data"
    exit 1
fi

../../build/tools/caffe train \
    -solver lrcn_solver.from_lm.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
