import json
import numpy as np
import pdb

def open_txt(f):
  open_f = open(f).readlines()
  return [f.strip() for f in open_f]

def read_json(t_file):
  j_file = open(t_file).read()
  return json.loads(j_file)

def save_json(json_dict, save_name):
  with open(save_name, 'w') as outfile:
    json.dump(json_dict, outfile)

def mat_to_one_hot(mat, max_value):
  mat_one_hot = np.zeros((mat.shape[0], mat.shape[1], max_value))
  a1_idx = [[i]*mat.shape[1] for i in range(mat.shape[0])]
  a1_idx = [i for j in a1_idx for i in j]
  a2_idx = range(mat.shape[1])*mat.shape[0]
  mat_one_hot[a1_idx, a2_idx, np.ndarray.flatten(mat)] = 1
  return mat_one_hot

def label_mat(mat):
 return np.argsort(mat, axis=1) 

def read_data(txt_data):
  data = open_txt(txt_data)
  data = [d.split(' ') for d in data]
  data = [[int(d) for d in data_list] for data_list in data]
  return np.array(data).astype(int)
