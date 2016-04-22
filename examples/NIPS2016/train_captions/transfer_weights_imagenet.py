import sys
sys.path.insert(0,'../../python/')
sys.path.insert(0, '../word_similarity/')
from w2vDist import *
import caffe
import numpy as np
import copy
import pickle as pkl
import hickle as hkl
from nltk.corpus import wordnet as wn
#import find_close_words

save_tag = 'closest_W2V_embedW2V'
transfer_embed = False 
num_close_words_im = 1
num_close_words_lm = 1

all_add_words = []

attributes_list = open('utils_trainAttributes/lexicalList_471_rebuttalScale.txt', 'r')
attributes_list = [a.strip() for a in attributes_list]
vocab_file = '../coco_caption/h5_data/buffer_100/vocabulary.txt'
vocab_lines = open(vocab_file, 'rb').readlines()
vocab_lines = [v.strip() for v in vocab_lines]
vocab_lines = ['<EOS>'] + vocab_lines

attributes_known = list(set(attributes_list) & set(vocab_lines))
attributes_unknown_vocab = list(set(attributes_list) - set(vocab_lines)) 

vocab_file = '../coco_caption/h5_data/buffer_100/vocabulary80k.txt'
vocab_lines = open(vocab_file, 'rb').readlines()
vocab_lines = [v.strip() for v in vocab_lines]
vocab_lines = ['<EOS>'] + vocab_lines


attributes_unknown = []

#create sysnset stuff
lexical_synsets = [None]*len(attributes_known + attributes_unknown_vocab)
for ix, attribute in enumerate(attributes_known + attributes_unknown_vocab):
  if len(wn.synsets(attribute)) > 0:
    #lexical_synsets[ix] = filter(lambda x: 'n' == x.pos(), wn.synsets(attribute))  
    if len(filter(lambda x: 'n' == x.pos(), wn.synsets(attribute))) > 0:
      lexical_synsets[ix] = filter(lambda x: 'n' == x.pos(), wn.synsets(attribute))[0]

#LISA: This should be neater, but it will work for now
#create word to vec
W2V = w2v()
W2V.readVectors()
W2V.reduce_vectors(attributes_known + attributes_unknown_vocab, '-n')

def closeness_synset(new_word):
  new_word_synset = filter(lambda x: 'n' == x.pos(), wn.synsets(new_word))
  closeness = []
  for ls in lexical_synsets:
    sim_nws = 0
    for nws in new_word_synset:
      if ls:
        sim = 0
        for l in [ls]:
          sim += nws.path_similarity(l)
        sim_nws += sim
        #div = len(ls)
        div = 1
      else:
        sim_nws = -10000
        div = 1
    closeness.append(sim_nws/div)
  return closeness

def closeness_embedding(new_word):
  return W2V.findClosestWords(new_word)

def closeness_both(new_word):
  ce = closeness_embedding(new_word)
  ces = closeness_synset(new_word)
  ces_div =  [c/max(ces) for c in ces]
  return ce + np.array(ces_div)

closeness_metric = closeness_embedding

attributes_orig = open('utils_trainAttributes/lexicalList_parseCoco_JJ100_NN300_VB100.txt').readlines()
attributes_orig = [a.strip() for a in attributes_orig]

for i, a in enumerate(attributes_unknown_vocab):
  print i
  if a in vocab_lines:
    sims = closeness_metric(a)
    if not isinstance(sims, int):
      attributes_unknown.append(a)

attributes = attributes_known + attributes_unknown

def closeness_embedding_force(new_word):
  word_sims = np.ones((len(attributes),))*-1
  force_idx = attributes.index('giraffe')
  word_sims[force_idx] = 100
  return W2V.findClosestWords(new_word)
closeness_metric = closeness_embedding #change metric here

del W2V
W2V = w2v()
W2V.readVectors()
W2V.reduce_vectors(attributes, '-n')
del lexical_synsets
lexical_synsets = [None]*len(attributes)
for ix, attribute in enumerate(attributes):
  if len(wn.synsets(attribute)) > 0:
    #lexical_synsets[ix] = filter(lambda x: 'n' == x.pos(), wn.synsets(attribute))  
    if len(filter(lambda x: 'n' == x.pos(), wn.synsets(attribute))) > 0:
      lexical_synsets[ix] = filter(lambda x: 'n' == x.pos(), wn.synsets(attribute))[0]



add_words = {}
add_words['words'] = attributes_unknown
add_words['classifiers'] = attributes_unknown
#change here to play with illegal words
add_words['illegal_words'] = attributes_unknown + list((set(attributes_known) - set(attributes_orig)))
#add_words['illegal_words'] = attributes_unknown


model='mrnn_attributes_fc8.direct.from_features.wtd.80k.prototxt'

#set up network
close_words_im = {}
close_words_lm = {}
#model_weights = '/z/lisaanne/snapshots_caption_models/attributes_JJ100_NN300_VB100_newImages_captions_ftLMPretrain.surf_lr0.01_iter_120000.80k_iter_110000'
#model_weights='/z/lisaanne/snapshots_caption_models/attributes_JJ100_NN300_VB100_moreImages_captions_ftLMPretrain.surf_lr0.01_iter_120000.80k_1104_iter_100000'
#model_weights='/z/lisaanne/snapshots_caption_models/attributes_JJ100_NN300_VB100_moreImages_captions_ftLMPretrain.surf_lr0.01_iter_120000.80k_1104_iter_110000'
#model_weights = '/yy2/lisaanne/mrnn_direct/snapshots/attributes_moreImages_supp.direct_surf_lr0.01_iter_120000.80k_lrp01_1111_iter_65000'
model_weights = '/yy2/lisaanne/mrnn_direct/snapshots/attributes_JJ100_NN300_VB100_646Imagenet_0123.direct_surf_lr0.01_iter_120000_iter_110000'

net = caffe.Net(model, model_weights + '.caffemodel', caffe.TRAIN)

if 'predict-lm' in net.params.keys():
  predict_lm = 'predict-lm'
else:
  predict_lm = 'predict'

#weight transfer
similar_words = open('imagenet_parsing_utils/newWords_pairs_bothSym.txt', 'w')
for aw, word in enumerate(add_words['words']):
  close_words_im[word] = {}
  word_sims = closeness_metric(add_words['classifiers'][aw])
  for illegal_word in add_words['illegal_words']:
    illegal_idx = attributes.index(illegal_word)
    word_sims[illegal_idx] = -100000

  close_words_im[word] = {}
  close_words_lm[word] = {}

  close_words_im[word]['close_words'] = [attributes[np.argsort(word_sims)[-num_close_words_im]]]
  close_words_lm[word]['close_words'] = [attributes[np.argsort(word_sims)[-num_close_words_lm]]]
  #close_words_im[word]['close_words'] = ['giraffe']
  #close_words_lm[word]['close_words'] = ['giraffe']
  close_words_im[word]['weights'] = [1.]
  close_words_lm[word]['weights'] = [1.]
  print "Similar word for %s is %s.\n" %(word, attributes[np.argsort(word_sims)[-num_close_words_lm]])   
  similar_words.writelines('%s %s\n' %(word, attributes[np.argsort(word_sims)[-num_close_words_lm]]))
similar_words.close()
 
predict_weights_lm = copy.deepcopy(net.params[predict_lm][0].data)
predict_bias_lm = copy.deepcopy(net.params[predict_lm][1].data)
predict_weights_im = copy.deepcopy(net.params['predict-im'][0].data)
#predict_bias_im = copy.deepcopy(net.params['predict-im'][1].data)

for aw, add_word in enumerate(add_words['words']):
  if add_word == 'stapler':
    print 'stop point'
  add_word_idx = vocab_lines.index(add_word)
  attribute_loc = attributes_list.index(add_words['classifiers'][aw])
  transfer_weights_lm = np.ones((predict_weights_lm.shape[1],))*0
  transfer_bias_lm = 0
  transfer_weights_im = np.ones((predict_weights_im.shape[1],))*0

  for wi, close_word in enumerate(close_words_im[add_word]['close_words']):
    close_word_idx = vocab_lines.index(close_word)
    transfer_weights_im += net.params['predict-im'][0].data[close_word_idx,:]*close_words_im[add_word]['weights'][wi]
  
  #Take care of classifier cross terms
  for wi, close_word in enumerate(close_words_im[add_word]['close_words']): 
    close_word_idx = vocab_lines.index(close_word)
    close_word_attribute_loc = attributes_list.index(close_word)
    transfer_weights_im[attribute_loc] = net.params['predict-im'][0].data[close_word_idx, close_word_attribute_loc]
    transfer_weights_im[close_word_attribute_loc] = 0

  for wi, close_word in enumerate(close_words_lm[add_word]['close_words']):
    close_word_idx = vocab_lines.index(close_word)
    transfer_weights_lm += net.params[predict_lm][0].data[close_word_idx,:]*close_words_lm[add_word]['weights'][wi]
    transfer_bias_lm += net.params[predict_lm][1].data[close_word_idx]*close_words_lm[add_word]['weights'][wi]

  predict_weights_lm[add_word_idx,:] = transfer_weights_lm
  predict_bias_lm[add_word_idx] = transfer_bias_lm
  predict_weights_im[add_word_idx,:] = transfer_weights_im

  for wi, close_word in enumerate(close_words_lm[add_word]['close_words']):
    close_word_idx = vocab_lines.index(close_word)
    predict_weights_im[close_word_idx,attribute_loc] = 0 
   
net.params[predict_lm][0].data[...] = predict_weights_lm
net.params[predict_lm][1].data[...] = predict_bias_lm
net.params['predict-im'][0].data[...] = predict_weights_im
#net.params['predict-im'][1].data[...] = predict_bias_im
net.save('%s.%s.caffemodel' %(model_weights, save_tag))

print close_words_im
print close_words_lm
print 'Saved to: %s.%s.caffemodel' %(model_weights, save_tag) 
print 'Done.'
