# DLAV_DepthEstim

Estimation of depth using the GLPDepth code and the drivingstereo dataset


#### Training
```
$ python ./code/train.py --dataset DrivingStereo --data_path ../../../work/scitas-share/datasets/Vita/civil-459/ --max_depth 80.0 --max_depth_eval 80.0 --crop_h 352 --crop_w 704
```


#### Testing
!!! Need weights .ckpt in folder ckpt
```
$ python ./code/test.py --dataset DrivingStereo --data_path ./datasets/ --ckpt_dir ./ckpt/????????.ckpt --do_evaluate  --max_depth 80.0 --max_depth_eval 80.0
```

If --do_evaluate --> returns metrics
If --save_visualize --> save depth maps


#### Activate environment
create environment https://scitas-data.epfl.ch/confluence/display/DOC/Python+Virtual+Environments
```
$ source venvs/depth_env/bin/activate
```