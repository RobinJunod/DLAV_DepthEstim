# Monocular Camera Depth Estimation 

Guessing depth from just one picture is really tricky as information about depth from a single image is ambiguous. That's because one picture doesn't give us enough clues about how far the objects are, usually depth estimation is easier done with a stereo camera. However, by using the precise depth labels obtained from the avaiable dataset, we provided our model with a robust ground truth against which to learn.
The inital code GLPDepth was firstly trained on nyu depth v2 which contains indoor images. Then it was trained on the kitti dataset and performs pretty well. Moreover, the GLPDepth code was reused to add some mask image modelling and achieve state of the art. As the kitti dataset is a small dataset, we trained it on the DrivingStereo dataset which is much larger and tunes some of the data augmenetation parameters.

## 0 GLP Depth

The model this project is based on is the GLP Depth model [^1]. This model was chosen due to his really good performence, it's simple structure as well as it's light weight. Even though the GLP model is not the n°1 state of the art, its structure has been retrieved by the paper "Dark Secrets of Masked Image Modeling" [^2] which is the current state of the art on the Kitti dataset.
![image](https://github.com/RobinJunod/DLAV_DepthEstim/assets/82818451/6f4cc877-c9f5-48f9-84a4-2024c6af3263)
The GLP model has a hierarchical transformer encoder to capture and convey the global context. Simultaneously, a lightweight yet powerful decoder generate an estimated depth map while considering local connectivity. 

## 1 Contributions
Two mains contributions were impleemnted to improve the model. First the model was trained on another much larger dataset. Second, multiple learning rate scheduler were implemented to find better minimizers. Additionally, L2-Regularization as well as data augmentation tuning were performed.

### 1.1 Data DrivingStereo

The model was trained using a big collection of examples from the DrivingStereo (DS) dataset. This dataset is huge, so it provided a lot for the model to learn from. After training, it was then put to the test with the KITTI dataset, which is another set of examples from a different collection. Usually, the DrivingStereo dataset is used for estimating depth using two cameras, kind of like how our two eyes work together to judge depth. But for this project, a different approach was taken. Only pictures from the left camera were used with the aim of guessing depth from a single image. This is quite a challenging task, but it's also really useful for things like self-driving cars and robots.


### 1.2 Learning Rate Scheduler

Several learning rate (LR) schedulers were implemented. The first one was the 'reduce on plateau' LR scheduler. Its job is to lower the learning rate by a certain amount whenever the model's loss stops getting smaller. This strategy can help the model learn quicker during the training stage. But one thing to note is that this scheduler is quite sensitive to the initial learning rate and to the hyperparameters.

The second scheduler used was the CyclicLR with 'triangle2' from the PyTorch library. This one alternates between a low and a high learning rate. The benefit of using higher learning rates is that it allows the model to explore different areas of the loss landscape. On the other hand, lower learning rates let the model dig deeper into the more promising areas. This flexibility can help find better solutions and prevent the model from getting stuck in non perfect optima. You can see how this scheduler works from the picture below.

![image](https://github.com/RobinJunod/DLAV_DepthEstim/assets/82818451/92f9e132-6059-4ec4-90bc-7dac600f88d2)


### 1.3 Other contributions

Some additional techniques were also used to make the model better. For example, L2 regularization was paired with the 'reduce on plateau' scheduler to help prevent overfitting. By using these technique, we're aiming to improve how well the model can generalize to new situations.

Some fine tuning were also made on the data augmentation part like random cropping. This is a type of data augmentation, which slightly altered the pictures to make the model more robust and able to handle a wider range of situations.

### 1.4 Problem faced

During the training phase on the entire dataset, we ran into an issue with our loss function. It was behaving well initially, reaching a good point, but then it unexpectedly exploded, ending up at a "NaN" values. This problem was solved by implementing gradient clipping, a technique that helps manage extreme changes in the loss function, keeping the training process more stable. 

## 2 Results
Below is the presentation of the results obtained using different learning rate (LR) schedulers during testing on the DrivingStereo dataset:

| Results on the DS  test dataset | delta1 | delta2 | delta3 | Abs. Rel. | Sq. Rel. | RMSE | RMSE Log |
|---------------------------------|--------|--------|--------|-----------|----------|------|----------|
| ReduceLROnPlateau               | 0.9735 | 0.9958 | 0.9989 | 0.0629  | 0.2808  |3.3034  | 0.0907  |
| ReduceLROnPlateau + L2 reg.     | 0.9580   |0.9928  |0.9978  | 0.0693  |0.3914  |3.8787  | 0.1054 |
| CycilcLR (triangle2)            |  0.9746   | 0.9960 |0.9989 | 0.0601   | 0.2733 |3.2714  |0.0890   |

In order to compare our results with the GLP Depth original best weights, we evaluated our model on the Kitti dataset. To ensure fairness, we also tested the best GLP model on the driving stereo dataset. The results indicate that our model demonstrated a better fit for the driving stereo testing images compared to the GLP best model. However, it performed worse when tested on the Kitti test data in comparison to the GLP best model.

|                               GLP Depth original                                 |
| Trained on | delta1 | delta2 | delta3 | Abs. Rel. | Sq. Rel. | RMSE | RMSE Log |
|------------|--------|--------|--------|-----------|----------|------|----------|
| Kitti             | 0.9735 | 0.9958 | 0.9989 | 0.0629  | 0.2808  |3.3034  | 0.0907  |
| Driving Stereo     | 0.9580   |0.9928  |0.9978  | 0.0693  |0.3914  |3.8787  | 0.1054 |


|                          GLP Depth with our contribution                  |
| Trained on | delta1 | delta2 | delta3 | Abs. Rel. | Sq. Rel. | RMSE | RMSE Log |
|------------|--------|--------|--------|-----------|----------|------|----------|
| Kitti             | 0.9735 | 0.9958 | 0.9989 | 0.0629  | 0.2808  |3.3034  | 0.0907  |
| Driving Stereo     | 0.9580   |0.9928  |0.9978  | 0.0693  |0.3914  |3.8787  | 0.1054 |

Below we can see the results of our model's improvments on a monocular video. This result can be done using the inference.py file.
![Comparaison of our result with the original one](https://github.com/RobinJunod/DLAV_DepthEstim/blob/main/result%20demo/result_DLAV_gif.gif)

## 3 Guide to use the model

### 3.1 Environment
The environment requierments are store in the .txt file and can be easly created using anaconda. To 
create an environment on scitas follow : https://scitas-data.epfl.ch/confluence/display/DOC/Python+Virtual+Environments
To activate the env on scitas execute this command
```
$ source venvs/<env name>/bin/activate
```
### 3.2 Training

For the trainnig part, some pretrained weight for the encoder can be loaded as a pre-training (mit_b4.pth). If they don't exist they will be downloaded automatically. 
```bash
  code/
  ├── models/
  │   ├── configs/  
  │   ├── dataset/  
  │   ├── weights/
  │   │   ├── mit_b4.pth/
  │   ├── utils/
```
  

Here below are the commands that should work using the avaiable training dataset on the scitas server.
```
$ python ./code/train.py --dataset DrivingStereo --data_path ../../../work/scitas-share/datasets/Vita/civil-459/ --max_depth 80.0 --max_depth_eval 80.0 --crop_h 352 --crop_w 704
```
```
$ python ./code/train.py --dataset DrivingStereo --data_path /work/scitas-share/datasets/Vita/civil-459/ --max_depth 80.0 --max_depth_eval 80.0 --workers 1 --lr 0.00015 --epochs 8 --save_model --workers 4 
```

##### list of usefull arguments for training :

| argument | type     | default  |
|----------|----------|----------|
|  --epochs  |  int   |   25     |
|  --lr     |  float  |  1e-4    |
|  --crop_h  |  int   |   448    |
|  --crop_w  |  int   |   576    |
|  --log_dir  |  str   |   ./logs|
|  --val_freq |  int   |   1    |
|  --save_freq  |  int   |   10|    

To save the model into a .ckpt (this can be tuned to save the weight at each epochs)
```
$ --save_model
```
save the results of the validation part (depth map from the model after every epochs)
```
$ --save_result 
```

### 3.3 Testing

For the testing part, the model weight must be stored in .ckpt in the folder of the same name.
```bash
  ckpt/
  ├── <model to test>.ckpt
  code/
  datasets/
```
Use this command by replacing the <model to test> by the actual name of the .ckpt model
```
$ python ./code/test.py --dataset DrivingStereo --data_path ./datasets/ --ckpt_dir ./ckpt/<model to test>.ckpt --do_evaluate  --max_depth 80.0 --max_depth_eval 80.0
```

returns metrics
```
--do_evaluate
```
save depth maps
```
--save_visualize 
```

## 4 Reference
 Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth
  
[^1]: Doyeon Kim1, Woonghyun, Ka Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth, [URL](https://github.com/vinvino02/GLPDepth)

[^2]:  Revealing the Dark Secrets of Masked Image Modeling , [URL] (https://github.com/SwinTransformer/MIM-Depth-Estimation)
  
