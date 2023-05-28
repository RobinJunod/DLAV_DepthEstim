import os

command = 'python ./code/test.py --dataset DrivingStereo --data_path ./datasets/ --ckpt_dir ./ckpt/best_model.ckpt --do_evaluate --max_depth 80.0 --max_depth_eval 80.0'

os.system(command)
