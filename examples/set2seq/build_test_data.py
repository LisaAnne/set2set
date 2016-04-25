import numpy as np
import random
import pdb
import sys
import math

def build_test_data(len_sequence=2, max_value=5):
  #determine how many unique sets I can get
  num_possible_sequences=math.factorial(max_value)/math.factorial(max_value-len_sequence)
  num_gen_sequences = min((1000000/len_sequence), num_possible_sequences)

  sequences = [None]*num_gen_sequences
  base_sequence = range(max_value)
  s_count = 0
  i_count = 0
  while s_count < num_gen_sequences and i_count < 1000000:
    i_count += 1
    if i_count % 100 == 0:
      print s_count, i_count
    random.shuffle(base_sequence)
    sequences[s_count] = tuple(base_sequence[:len_sequence])
    sequences_unique = list(set(sequences[:s_count+1]))
    s_count = len(sequences_unique)
 
  train_split_point = int(0.75*s_count)
  train_set = sequences_unique[:train_split_point]
  test_set = sequences_unique[train_split_point:]

  test_save_name = 'utils/data/ls_%d_mv_%d_test.txt' %(len_sequence, max_value)
  train_save_name = 'utils/data/ls_%d_mv_%d_train.txt' %(len_sequence, max_value)

  def save_set(save_name, save_set):
    save = open(save_name, 'w')
    for t in save_set:
      save.writelines('%s\n' %(' '.join(map(str, list(t)))))
    save.close()

  save_set(test_save_name, test_set)
  save_set(train_save_name, train_set)

if __name__ == '__main__':
  build_test_data(int(sys.argv[1]), int(sys.argv[2]))

