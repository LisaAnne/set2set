import sys
sys.path.insert(0,'../../python/')
import caffe
import copy
import pdb

proto = 'prototxts/deploy.coco_directfc7_voc72klabel_shared_glove.prototxt'
language_net = 'snapshots/coco_directfc7_voc72klabel_glove_sgd_iter_55000.caffemodel'
image_net = 'snapshots/coco_fc7_to_label_voc72k_sgdlre2_iter_60000.caffemodel.h5'

language_net = caffe.Net(proto, language_net, caffe.TRAIN)
image_net = caffe.Net(proto, image_net, caffe.TRAIN)
language_net.params['predict-im'][0].data[...] = copy.deepcopy(image_net.params['predict-im'][0].data)
pdb.set_trace()
language_net.save('/yy2/lisaanne/NIPS2016_newWords/extracted_features/coco_learnClassifier_voc72klabel_glove_sgd_iter_55000.caffemodel')
