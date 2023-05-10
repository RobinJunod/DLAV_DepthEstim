import os
import cv2

from dataset.base_dataset import BaseDataset


class drivingstereo(BaseDataset):
    def __init__(self, data_path, filenames_path='./code/dataset/filenames/', 
                 is_train=True, dataset='DrivingStereo', crop_size=(352, 704),
                 scale_size=(1216,352)):
        super().__init__(crop_size)        

        self.scale_size = scale_size
        
        self.is_train = is_train
        self.data_path = os.path.join(data_path, 'DrivingStereo') 

        self.image_path_list = []
        self.depth_path_list = []
        txt_path = os.path.join(filenames_path, 'DrivingStereo')
        
        if is_train:
            txt_path += '/drivingstereo_all_train.txt' # '/drivingstereo_all_train.txt' for scitas
        else:
            txt_path += '/drivingstereo_all_test.txt'        
        
        self.filenames_list = self.readTXT(txt_path)
        phase = 'train' if is_train else 'test'        
        print("Dataset :", dataset)
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    # kb cropping
    def cropping(self, img):
        h_im, w_im = img.shape[:2]

        margin_top = int(h_im - 352)
        margin_left = int((w_im - 1216) / 2)

        img = img[margin_top: margin_top + 352,
                  margin_left: margin_left + 1216]

        return img

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = self.data_path + self.filenames_list[idx].split(' ')[0] # 0 = left image / 1 = right image
        gt_path = self.data_path + self.filenames_list[idx].split(' ')[2] # 2 for depth map
        filename = img_path.split('/')[-4] + '_' + img_path.split('/')[-1]

        image = cv2.imread(img_path)  # [H x W x C] and C: BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
        
        image = self.cropping(image)
        depth = self.cropping(depth)

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(depth, (self.scale_size[0], self.scale_size[1]))

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        depth = depth / 256.0  # convert in meters (same as Kitti)

        return {'image': image, 'depth': depth, 'filename': filename}
