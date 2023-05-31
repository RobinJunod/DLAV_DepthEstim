'''
Doyeon Kim, 2022
'''

import os
import cv2
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import utils.logging as logging
import utils.metrics as metrics
from models.model import GLPDepth
from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


def main():
    # experiments setting
    opt = TestOptions()
    args = opt.initialize().parse_args()
    print(args)

    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')


    result_path = os.path.join(args.result_dir, args.exp_name)
    logging.check_and_make_dirs(result_path)
    print("Saving result images in to %s" % result_path)
    

    print("\n1. Define Model")
    model = GLPDepth(max_depth=args.max_depth, is_train=False).to(device)
    model_weight = torch.load(args.ckpt_dir)
    #model_weight = torch.load(args.ckpt_dir, map_location=torch.device('cpu')) # load weight for CPU
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()

    print("\n2. Define Dataloader")
    dataset_kwargs = {'dataset_name': 'ImagePath', 'data_path': args.data_path}


    test_dataset = get_dataset(**dataset_kwargs)
    print(dataset_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True)

    print("\n3. Inference & Evaluate")
    for batch_idx, batch in enumerate(test_loader):
        input_RGB = batch['image'].to(device)
        filename = batch['filename']

        with torch.no_grad():
            pred = model(input_RGB)
        pred_d = pred['pred_d']


        save_path = os.path.join(result_path, filename[0])
        pred_d_numpy = pred_d.squeeze().cpu().numpy()
        pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
        pred_d_numpy = pred_d_numpy.astype(np.uint8)
        pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
        cv2.imwrite(save_path, pred_d_color)
        logging.progress_bar(batch_idx, len(test_loader), 1, 1)


    print("Done")


if __name__ == "__main__":
    main()
