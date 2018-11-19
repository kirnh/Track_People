from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)

if cuda:
    model.cuda()

model.eval() # Set in evaluation mode

dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

classes = load_classes(opt.class_path) # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print ('\nPerforming object detection on:')

prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))
    # print('Input shape:')
    # print(input_imgs.shape)

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)


    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    image = cv2.imread(img_paths[0])
    # print(image.shape)
    # print(input_imgs.shape[1:])
    img_detections.extend(detections)
    # print('--------------------')
    # print(img_paths[0])
    # count = 0

    # The amount of padding that was added
    pad_x = max(image.shape[0] - image.shape[1], 0) * (opt.img_size / max(image.shape))
    pad_y = max(image.shape[1] - image.shape[0], 0) * (opt.img_size / max(image.shape))
    # Image height and width after padding is removed
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x
    bboxes = []
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
        # Rescale coordinates to original dimensions
        box_h = ((y2 - y1) / unpad_h) * image.shape[0]
        box_w = ((x2 - x1) / unpad_w) * image.shape[1]
        y1 = ((y1 - pad_y // 2) / unpad_h) * image.shape[0]
        x1 = ((x1 - pad_x // 2) / unpad_w) * image.shape[1]
        # convert to top left, bottom right
        bbox = np.array([x1, y1, box_w, box_h]) 
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            continue
        sx, sy, ex, ey = bbox
        bboxes.append(bbox)
        # patch_image = image[sy:ey, sx:ex]
        # print('Image shape')
        # print(image.shape)
        # print('Patch shape')
        # print(patch_image.shape)
        # patch_image = cv2.resize(image, tuple(patch_shape[::-1]))
        # cv2.imwrite('{}.jpeg'.format(count), patch_image)
        # print('wrote '+'{}.jpeg'.format(count))
        # count+=1

        # Need to return bbox, confidence, feature (128 length vector coming from another network) for tracking app

# Below code used to plot and save images


# # print('----')
# # print(imgs)
# # Bounding-box colors
# cmap = plt.get_cmap('tab20b')
# colors = [cmap(i) for i in np.linspace(0, 1, 20)]

# print ('\nSaving images:')
# # Iterate through images and save plot of detections
# for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

#     print ("(%d) Image: '%s'" % (img_i, path))

#     # Create plot
#     img = np.array(Image.open(path))
#     plt.figure()
#     fig, ax = plt.subplots(1)
#     ax.imshow(img)

#     # The amount of padding that was added
#     pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
#     pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
#     # Image height and width after padding is removed
#     unpad_h = opt.img_size - pad_y
#     unpad_w = opt.img_size - pad_x

#     # Draw bounding boxes and labels of detections
#     if detections is not None:
#         unique_labels = detections[:, -1].cpu().unique()
#         n_cls_preds = len(unique_labels)
#         bbox_colors = random.sample(colors, n_cls_preds)
#         for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

#             print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

#             # Rescale coordinates to original dimensions
#             box_h = ((y2 - y1) / unpad_h) * img.shape[0]
#             box_w = ((x2 - x1) / unpad_w) * img.shape[1]
#             y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
#             x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

#             color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
#             # Create a Rectangle patch
#             bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
#                                     edgecolor=color,
#                                     facecolor='none')
#             # Add the bbox to the plot
#             ax.add_patch(bbox)
#             # Add label
#             plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
#                     bbox={'color': color, 'pad': 0})

#     # Save generated image with detections
#     plt.axis('off')
#     plt.gca().xaxis.set_major_locator(NullLocator())
#     plt.gca().yaxis.set_major_locator(NullLocator())
#     print('output/%d.png' % (img_i))
#     plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
#     plt.close()
