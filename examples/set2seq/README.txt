Set to Set:

1.	First run setup.sh 

2.	Run test_model.py
test_model.py has the following parameters:
	len_sequence: length of sequence to sort
	message_dim: dimension of message vector
	process_steps: number of steps in process block
	num_tests: number of different network initializations to try 

Example: python test_model.py --len_sequence 10 --message_dim 10 --process_steps 1 --num_tests 5



