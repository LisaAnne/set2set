#!/usr/bin/env bash

#coco
deploy_image=dcc_vgg.471.deploy.prototxt
deploy_words=deploy.coco_directfc7_voc72klabel_shared_glove.prototxt
#jointly learn classifier
#model_name=coco_directfc7_voc72klabel_glove_sgd_iter_55000.caffemodel
#pretrained features
#model_name=coco_learnClassifier_voc72klabel_glove_sgd_iter_55000.caffemodel
#model_name=coco_directfc7_voc72klabel_glove_sgd_iter_55000.caffemodel
#model_name=coco_directfc7_fixIM_voc72klabel_glove_sgd_iter_45000.caffemodel
#model_name=coco_directfc7_fixIM_voc72klabel_glove_sgd_lr4e4_iter_50000.caffemodel

#tanh model
deploy_words=tansum_coco_fc7_voc72klabel.shared_glove72k.prototxt
model_name=tansum_coco_directfc7_fixIM_voc72klabel_glove_sgd_lr4e4_iter_55000.caffemodel

vocab=surf_intersect_glove.txt
precomputed_feats='/yy2/lisaanne/vgg_features/h5Files/coco2014_cocoid.val_val.txt0_fullLabel2.h5'
image_list=coco2014_cocoid.val_val.txt
split=val_val
language_feature=probs

echo $deploy_image
echo $deploy_words
echo $model_name
echo $vocab
echo $precomputed_feats
echo $image_list

python dcc.py --image_model $deploy_image \
              --language_model $deploy_words \
              --model_weights $model_name \
              --vocab $vocab \
              --precomputed_features $precomputed_feats \
              --image_list $image_list \
              --split $split \
              --language_feature $language_feature \
              --generate_coco
