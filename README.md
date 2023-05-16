# DLAV_DepthEstim
Estimation of depth using the GLPDepth code and the drivingstereo dataset.
The inital code GLPDepth was firstly trained on nyudepth which contains indoor imagers. Then it was trained on the kitti dataset and performs pretty well. Moreover, the GLPDepth code was reused to add some mask image modelling and achieve state of the art. As the kitti dataset is a small dataset, we trained it on the DrivingStereo dataset which is much larger and tunes some of the dataaugmenetation parameters.
Description of the dataset, label format, where/how to acquire it.

What data do I need
to train your model? How do I get it? In what shape?


#### Training
```
$ python ./code/train.py --dataset DrivingStereo --data_path ../../../work/scitas-share/datasets/Vita/civil-459/ --max_depth 80.0 --max_depth_eval 80.0 --crop_h 352 --crop_w 704
```
```
$ python ./code/train.py --dataset DrivingStereo --data_path /work/scitas-share/datasets/Vita/civil-459/ --max_depth 80.0 --max_depth_eval 80.0 --workers 1 --lr 0.00015 --epochs 8 --save_model --workers 4 
```


#### Testing
!!! Need weights .ckpt inside the folder ckpt
```
$ python ./code/test.py --dataset DrivingStereo --data_path ./datasets/ --ckpt_dir ./ckpt/????????.ckpt --do_evaluate  --max_depth 80.0 --max_depth_eval 80.0
```

list of usefull arguments for training :
--epochs,     type=int,   default=25)
--lr,         type=float, default=1e-4)
--crop_h,     type=int,   default=448)
--crop_w,     type=int,   default=576)        
--log_dir,    type=str,   default='./logs')
--val_freq, type=int, default=1)
--save_freq, type=int, default=10)
--save_model  --> save the model into a .ckpt for every 10 epochs      
--save_result --> save the results of the validation part (depth map from the model after every epochs)

list of usefull arguments for testing :
--do_evaluate --> returns metrics
--save_visualize --> save depth maps


#### Activate environment
create environment https://scitas-data.epfl.ch/confluence/display/DOC/Python+Virtual+Environments
Install all the packages in the requierments.txt 
```
$ source venvs/depth_env/bin/activate
```
