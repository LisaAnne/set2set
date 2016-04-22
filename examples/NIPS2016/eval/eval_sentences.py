from utils.config import *
from utils.python_utils import *
import sys
import pdb
import re
import numpy as np
import os

COCO_PATH = caffe_dir + '/data/coco/coco-caption/'
sys.path.insert(0,COCO_PATH)
from pycocotools.coco import COCO

COCO_EVAL_PATH = caffe_dir + '/data/coco/coco-caption-eval/'
sys.path.insert(0,COCO_EVAL_PATH)
from pycocoevalcap.eval import COCOEvalCap

rm_word_dict = {'bus': ['bus', 'busses'],
                'bottle': ['bottle', 'bottles'],
                'couch': ['couch', 'couches', 'sofa', 'sofas'],
                'microwave': ['microwave', 'microwaves'],
                'pizza': ['pizza', 'pizzas'],
                'racket': ['racket', 'rackets', 'racquet', 'racquets'],
                'suitcase': ['luggage', 'luggages', 'suitcase', 'suitcases'],
                'zebra': ['zebra', 'zebras']} 

def split_sent(sent):
  sent = sent.lower()
  sent = re.sub('[^A-Za-z0-9\s]+','', sent)
  return sent.split()

def score_generation(gt_filename=None, generation_result=None):

  coco_dict = read_json(generation_result)


  coco = COCO(gt_filename)
  generation_coco = coco.loadRes(generation_result)
  coco_evaluator = COCOEvalCap(coco, generation_coco)
  #coco_image_ids = [self.sg.image_path_to_id[image_path]
  #                  for image_path in self.images]
  coco_image_ids = [j['image_id'] for j in coco_dict]
  coco_evaluator.params['image_id'] = coco_image_ids
  results = coco_evaluator.evaluate(return_results=True)
  return results

def F1(generated_json, novel_ids, train_ids, word):
  set_rm_words = set(rm_word_dict[word])
  gen_dict = {}
  for c in generated_json:
    gen_dict[c['image_id']] = c['caption']

  #true positive are sentences that contain match words and should
  tp = sum([1 for c in novel_ids if len(set_rm_words.intersection(set(split_sent(gen_dict[c])))) > 0]) 
  #false positive are sentences that contain match words and should not
  fp = sum([1 for c in train_ids if len(set_rm_words.intersection(set(split_sent(gen_dict[c])))) > 0])
  #false positive are sentences that do not contain match words and should
  fn = sum([1 for c in novel_ids if len(set_rm_words.intersection(set(split_sent(gen_dict[c])))) == 0 ])
 
  #precision = tp/(tp+fp)
  if tp > 0:  
    precision = float(tp)/(tp+fp) 
    #recall = tp/(tp+fn)
    recall = float(tp)/(tp+fn)
    #f1 = 2* (precision*recall)/(precision+recall)
    return 2*(precision*recall)/(precision+recall)
  else:
    return 0.

def score_result_subset(result, ids, metrics=['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']):

  metric_dict = {}
  for m in metrics: metric_dict[m] = []

  for id in ids:
    for m in metrics:
      metric_dict[m].append(result[id][m])
  for m in metric_dict: metric_dict[m] = np.mean(metric_dict[m])

  for m in metrics:
    print "%s: %.04f" %(m, metric_dict[m])

def make_imagenet_result_dict(imagenet_sents):
  caps = read_json(imagenet_sents)

  imagenet_result_dict = {}
  for cap in caps:
    key = cap['image_id'].split('/')[0]
    if key not in imagenet_result_dict.keys():
      imagenet_result_dict[key] = {}

    imagenet_result_dict[key][cap['image_id']] = cap['caption']

  return imagenet_result_dict

def make_imagenet_html(imagenet_result_dict, imagenet_result_dict_baseline, base='w2v_transfer'):
  webpage_base='../../../output_html/cvpr2016/'
  real_root='/yy2/lisaanne/imageData/imagenet_rebuttal/'
  for o in imagenet_result_dict.keys():
    f = open('%s%s_%s.html' %(webpage_base, base, o), 'w')
    message = ''
    message += '<table style="width:100%" table border="1" rules="rows,columns">'
    for ix, im_id in enumerate(imagenet_result_dict[o].keys()):
      sym_path = '%s/web_images/%s' %(webpage_base, im_id)
      sym_folder = '%s/web_images/%s' %(webpage_base, im_id.split('/')[0])
      if not os.path.isdir(sym_folder):
        os.mkdir(sym_folder)
      if not os.path.isfile(sym_path):
        os.symlink(real_root+im_id, sym_path)
      message += '<tr>'
      message += '<td>'
      message += '</br>'
      message += '<IMG SRC="%s" width="250"><br/>\n' %('web_images/' + im_id)
      message += '</td>'
      message += '<td>'
      message += 'Results for image %d (%s) <br/>\n' %(ix+1, im_id.split('/')[-1])
      message += '</br>'
      message += 'Transfer caption: %s</br>\n' %imagenet_result_dict[o][im_id]
      message += 'Baseline caption: %s</br>\n' %imagenet_result_dict_baseline[o][im_id]
      message += '</br>'
      message += '</td>'
      message += '</tr>'
    message += '</table>'
    message += '<br/><br/>'
    f.writelines(message)
    print "Wrote %s." %('%s%s_%s.html' %(webpage_base, base, o))
    f.close()

def find_successful_classes(imagenet_result_dict, num_train_examples=100):

  #build num_train_examples dict
  train_images = open_txt('utils/imageList/train_images_rebuttal.txt')
  train_count = {}
  for line in train_images:
    o = line.split('/')[0]
    if o not in train_count.keys():
      train_count[o] = 0
    train_count[o] += 1

  successful_class = 0
  total_class = 0
  for o in train_count.keys():
    if train_count[o] >= num_train_examples:
      count_caps = 0
      total_class += 1
      for im_id in imagenet_result_dict[o].keys():
        sent = split_sent(imagenet_result_dict[o][im_id])
        if o in sent:
          count_caps += 1
      if count_caps > 0:
        successful_class += 1
  print "Percent successful classes: %f" %(float(successful_class)/total_class)
  print "Correct classes: %d" %successful_class
  print "Total classes: %d" %total_class

def add_new_word(gt_filename, generation_result, words, dset_name='val_val'):
  results = score_generation(gt_filename, generation_result)
  generation_sentences = read_json(generation_result)
  for word in words:
    gt_novel_file = annotations + 'captions_split_set_%s_%s_novel2014.json' %(word, dset_name)
    gt_train_file = annotations + 'captions_split_set_%s_%s_train2014.json' %(word, dset_name)
    gt_novel_json = read_json(gt_novel_file)
    gt_train_json = read_json(gt_train_file)
   
    gt_novel_ids = [c['image_id'] for c in gt_novel_json['annotations']]
    gt_train_ids = [c['image_id'] for c in gt_train_json['annotations']]
 
    gen_novel = [] 
    gen_train = []
    for c in  generation_sentences:
      if c['image_id'] in gt_novel_ids:
        gen_novel.append(c)
      else: 
        gen_train.append(c)

    save_json(gen_novel, 'tmp_gen_novel.json')
    save_json(gen_train, 'tmp_gen_train.json')

    print "Word: %s.  Novel scores:" %word
    score_generation(gt_novel_file, 'tmp_gen_novel.json')
    print "Word: %s.  Train scores:" %word
    score_generation(gt_train_file, 'tmp_gen_train.json')
    f1 =  F1(generation_sentences, gt_novel_ids, gt_train_ids, word)
    print "Word: %s.  F1 score: %.04f\n" %(word, f1)

    os.remove('tmp_gen_novel.json')
    os.remove('tmp_gen_train.json')

def make_coco_html(base='coco_examples'):
  generated_sentences = 'results/generated_sentences/'
  coco_baseline = read_json(generated_sentences + 'vgg_471Objects_coco_baseline_iter_110000_val_val_beam1_coco.json')
  coco_noT = read_json(generated_sentences + '_coco_rm1_vgg.471.solver.prototxt_iter_110000_val_val_beam1_coco.json')
  coco_T = read_json(generated_sentences + '_coco_rm1_vgg.471.solver.prototxt_iter_110000.transfer_words_coco1.txt_closeness_embedding_val_val_beam1_coco.json')
  coco_ood = read_json(generated_sentences + '_oodLM_rm1_vgg.im2txt.471.solver.prototxt_iter_110000.transfer_words_coco1.txt_closeness_embedding_val_val_beam1_coco.json')

  coco_result_dict = {}
  for a in coco_baseline:
    im_id = a['image_id']
    coco_result_dict[im_id] = {}
    coco_result_dict[im_id]['baseline'] = a['caption']
  for a in coco_noT:
    im_id = a['image_id']
    coco_result_dict[im_id]['coco_noTransfer'] = a['caption']
  for a in coco_T:
    im_id = a['image_id']
    coco_result_dict[im_id]['coco_transfer'] = a['caption']
  for a in coco_ood:
    im_id = a['image_id']
    coco_result_dict[im_id]['coco_ood'] = a['caption'] 

  webpage_base='../../../output_html/cvpr2016/'
  real_root='/home/lisaanne/caffe-LSTM/data/coco/coco/images/val2014/COCO_val2014_%012d.jpg'
  #for ix, im_id in enumerate(coco_result_dict.keys()):
  for ix in range(0, len(coco_result_dict.keys()),1000):
    f = open('%s%s_%d.html' %(webpage_base, base, ix), 'w')
    message = ''
    message += '<table style="width:100%" table border="1" rules="rows,columns">'
    for i in range(ix, min(ix+1000, len(coco_result_dict.keys()))):
      im_id = coco_result_dict.keys()[i]
      sym_path = '%s/web_images/coco/COCO_val2014_%012d.jpg' %(webpage_base, im_id)
      sym_folder = '%s/web_images/coco' %webpage_base
      if not os.path.isdir(sym_folder):
        os.mkdir(sym_folder)
      if not os.path.isfile(sym_path):
        os.symlink(real_root %im_id, sym_path)
      message += '<tr>'
      message += '<td>'
      message += '</br>'
      message += '<IMG SRC="web_images/coco/COCO_val2014_%012d.jpg" width="250"><br/>\n' %(im_id)
      message += '</td>'
      message += '<td>'
      message += 'Results for image %d (%s) <br/>\n' %(i+1, im_id)
      message += '</br>'
      message += 'Pair supervision: %s</br>\n' %coco_result_dict[im_id]['baseline']
      message += 'No transfer: %s</br>\n' %coco_result_dict[im_id]['coco_noTransfer']
      message += 'DCC (in): %s</br>\n' %coco_result_dict[im_id]['coco_transfer']
      message += 'DCC (out): %s</br>\n' %coco_result_dict[im_id]['coco_ood']
      message += '</br>'
      message += '</td>'
      message += '</tr>'
    message += '</table>'
    message += '<br/><br/>'
    f.writelines(message)
    print "Wrote %s." %('%s%s_%d.html' %(webpage_base, base, ix))
    f.close()




