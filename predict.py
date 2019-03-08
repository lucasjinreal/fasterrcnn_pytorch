# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from lib.model.roi_layers import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.utils.net_utils import save_net, load_net, vis_detections
from lib.model.faster_rcnn.vgg16 import vgg16
from lib.model.faster_rcnn.resnet import resnet
from lib.model.utils.blob import im_list_to_blob

import pdb
from alfred.dl.torch.common import device
import glob
import cv2

net = 'vgg16'
# for VOC
all_classes = np.asarray(['__background__',
                          'aeroplane', 'bicycle', 'bird', 'boat',
                          'bottle', 'bus', 'car', 'cat', 'chair',
                          'cow', 'diningtable', 'dog', 'horse',
                          'motorbike', 'person', 'pottedplant',
                          'sheep', 'sofa', 'train', 'tvmonitor'])
num_classes = len(all_classes)
class_agnostic = False
checkpoint_path = os.path.join(
    './checkpoints', 'faster_rcnn_vgg16_voc_like.pth')
img_dir = './images'

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


if __name__ == '__main__':
    cfg_from_list(['ANCHOR_SCALES', '[8, 16, 32]',
                   'ANCHOR_RATIOS', '[0.5,1,2]'])
    cfg.TRAIN.USE_FLIPPED = False
    if net == 'vgg16':
        fasterRCNN = vgg16(all_classes, pretrained=False,
                           class_agnostic=class_agnostic)
    elif net == 'res101':
        fasterRCNN = resnet(all_classes, 101, pretrained=False,
                            class_agnostic=class_agnostic)
    elif net == 'res50':
        fasterRCNN = resnet(all_classes, 50, pretrained=False,
                            class_agnostic=class_agnostic)
    elif net == 'res152':
        fasterRCNN = resnet(all_classes, 152, pretrained=False,
                            class_agnostic=class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()
    fasterRCNN.to(device)
    checkpoint = torch.load(checkpoint_path)
    fasterRCNN.load_state_dict(checkpoint['state_dict'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1).to(device)
    im_info = torch.FloatTensor(1).to(device)
    num_boxes = torch.LongTensor(1).to(device)
    gt_boxes = torch.FloatTensor(1).to(device)

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if torch.cuda.is_available():
        cfg.CUDA = True

    start = time.time()
    max_per_image = 100
    thresh = 0.05
    fasterRCNN.eval()
    all_images_f = glob.glob(os.path.join(img_dir, '*.jpg'))
    print('predict on all {} images'.format(len(all_images_f)))

    for img_f in all_images_f:
        im_in = cv2.imread(img_f)
        im = im_in[:, :, ::-1]
        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data[0]
        boxes = rois.data[:, :, 1:5][0]

        for j in range(1, num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # print(inds)
            # print(inds.numel())
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                # if class_agnostic:
                #     cls_boxes = boxes[inds, :]
                # else:
                #     cls_boxes = boxes[inds][:, j * 4:(j + 1) * 4]
                cls_boxes = boxes[inds, :]
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :],
                           cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                im_in = vis_detections(
                    im_in, all_classes[j], cls_dets.cpu().numpy(), 0.5)
        cv2.imshow('result', im_in)
        cv2.waitKey(0)
