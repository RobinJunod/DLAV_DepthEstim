# DLAV_DepthEstim
Estimation of depth using the GLPDepth code and the drivingstereo dataset.
The inital code GLPDepth was firstly trained on nyudepth which contains indoor imagers. Then it was trained on the kitti dataset and performs pretty well. Moreover, the GLPDepth code was reused to add some mask image modelling and achieve state of the art. As the kitti dataset is a small dataset, we trained it on the DrivingStereo dataset which is much larger and tunes some of the dataaugmenetation parameters.
Description of the dataset, label format, where/how to acquire it.

### Contributions
Two mains contributions were impleemnted to improve the model. First the model was trained on another much larger dataset. Second, multiple learning rate scheduler were implemented to find better minimizers. Additionally, L2-Regularization as well as data augmentation tuning were performed.

#### Data DrivingStereo
The model was trained using the Driving stero dataset in odrer to take advantage of the large number of sample this dataset has. Then it was tested on the kitti dataset. The DrivingStereo dataset is originally used for depth estimation using 2 cameras. In this project, only the left camera images were used to achieve monocular depth estimation.

#### Data DrivingStereo

The Driving stero dataset contains more than 170'000 training sample, yet it is not used for monocular depth estinmation. For this reason, we trained the model on it to hopefully get some better results. 

#### Learning Rate Scheduler
Multiple LR scheduler were implemented. First the 'reduce on plateau' LR scheduler. This one deacrease the LR by a certain factor each time the loss stop deacreasing. By doing that the model will achieve faster convergence during the training process. However, this scheduler is sensitive to the initial learning rate and other hyperparameters. The second Scheduler implemented was a 

#### Other contributions


#### Problem faced
We had during training on the whole dataset issue with our loss that went to a good point an then exploded to finally be nan value. To solve this issue gradiant clipping was implemented. 


#### Training
```
$ python ./code/train.py --dataset DrivingStereo --data_path ../../../work/scitas-share/datasets/Vita/civil-459/ --max_depth 80.0 --max_depth_eval 80.0 --crop_h 352 --crop_w 704
```
```
$ python ./code/train.py --dataset DrivingStereo --data_path /work/scitas-share/datasets/Vita/civil-459/ --max_depth 80.0 --max_depth_eval 80.0 --workers 1 --lr 0.00015 --epochs 8 --save_model --workers 4 
```


#### Testing

Need weights .ckpt inside the folder ckpt
```
$ python ./code/test.py --dataset DrivingStereo --data_path ./datasets/ --ckpt_dir ./ckpt/????????.ckpt --do_evaluate  --max_depth 80.0 --max_depth_eval 80.0
```

list of usefull arguments for training :

| argument | type     | default   
|----------|----------|----------|
|  --epochs  |  int   |   25     |
|  --lr     |  float  |  1e-4    |
|  --crop_h  |  int   |   448    |
|  --crop_w  |  int   |   576    |
|  --log_dir  |  str   |   ./logs|
|  --val_freq |  int   |   1    |
|  --save_freq  |  int   |   10|    
```
$ --save_model  #save the model into a .ckpt for every 10 epochs      
$ --save_result  #save the results of the validation part (depth map from the model after every epochs)
```
list of usefull arguments for testing :
--do_evaluate --> returns metrics
--save_visualize --> save depth maps


#### Activate environment
create environment https://scitas-data.epfl.ch/confluence/display/DOC/Python+Virtual+Environments
Install all the packages in the requierments.txt 
```
$ source venvs/depth_env/bin/activate
```
