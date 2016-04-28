import build_net.caffe_net as cn
import build_net.set2seq_net as set2seq
import argparse

def build_set2seq(len_sequence, batch_size, max_value, message_dim, process_steps):

  tag = 'set2seq_bs%d_ls%d_md%d_ps%d' %(batch_size, len_sequence, message_dim, process_steps), 
  train_net = '%s_train.prototxt' %tag
  solver_net = '%s_solver.prototxt' %tag
  bash = 'run_%s.sh' %tag

  param_str = {'len_sequence': len_sequence, 'batch_size': batch_size,
               'max_value': max_value, 
               'top_names': ['rand_data', 'label_data', 'train_label_data'], 
               'message_dim': message_dim, 'process_steps': process_steps}

  net = set2seq.set2sequence_net(param_str)
  net.build_set2seq(param_str, train_net)   
  solver_params = {'max_iter': 10000, 'stepsize': 5000, 'snapshot': 10000, 
                   'base_lr': 0.01}
  cn.make_solver(solver_net, train_net, [], **solver_params)
  #cn.make_bash_script(bash, solver_net)
  return tag

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

#good params for len_sequence = 5: batch_size=100, message_dim=10, process steps = 1

  parser.add_argument("--len_sequence", type=int, default=10)
  parser.add_argument("--max_value", type=int, default=1)
  parser.add_argument("--batch_size", type=int, default=100)
  parser.add_argument("--message_dim", type=int, default=10)
  parser.add_argument("--process_steps", type=int, default=5)

  args = parser.parse_args()

  build_set2seq(args.len_sequence, args.batch_size, args.max_value, args.message_dim,
                args.process_steps)
