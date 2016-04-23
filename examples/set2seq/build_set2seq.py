import build_net.caffe_net as cn
import build_net.set2seq_net as set2seq
import argparse

def build_set2seq(len_sequence, batch_size, max_value):

  tag = 'set2seq'
  train_net = '%s_train.prototxt' %tag
  solver_net = '%s_solver.prototxt' %tag
  bash = 'run_%s.sh' %tag

  param_str = {'len_sequence': len_sequence, 'batch_size': batch_size,
               'max_value': max_value, 'top_names': ['rand_data', 'label_data']}

  net = set2seq.set2sequence_net(param_str)
  net.build_set2seq(param_str, train_net)   
  solver_params = {'max_iter': 100, 'stepsize': 50, 'snapshot': 100}
  cn.make_solver(solver_net, train_net, [], **solver_params)
  cn.make_bash_script(bash, solver_net)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--len_sequence", type=int, default=2)
  parser.add_argument("--max_value", type=int, default=5)
  parser.add_argument("--batch_size", type=int, default=50)

  args = parser.parse_args()

  build_set2seq(args.len_sequence, args.batch_size, args.max_value)
