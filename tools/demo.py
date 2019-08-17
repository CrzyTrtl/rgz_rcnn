#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the MIT license
#
#    Created on 15 March 2018 by chen.wu@icrar.org
#    
#    Modified by Vinay Kerai (22465472@student.uwa.edu.au) 

import os, sys, time
import os.path as osp
import argparse
from itertools import cycle

import xml.etree.ElementTree as ET
import csv
import pandas as pd

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from astropy.io import fits
import tensorflow as tf

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect, remove_embedded
from fast_rcnn.nms_wrapper import nms
from networks.factory import get_network
from utils.timer import Timer
from download_data import get_rgz_root
from fuse_radio_ir import fuse

CLASSES =  ('__background__', # always index 0
                            '1_1', '1_2', '1_3', '2_2', '2_3', '3_3')

colors_ = cycle(['cyan', 'yellow', 'magenta'])
colors_2 = ['cyan', 'yellow', 'magenta', 'green', 'blue', 'orange', 'darkviolet', 'springgreen', 'teal', 'hotpink', 'tab:brown']


#Set this to be the path to rgz_rcnn folder
path_U = os.path.dirname(os.getcwd())

ground_truth = {'_image':[], 'act_class': [], 'act_xmin':[], 'act_ymin':[], 'act_xmax':[], 'act_ymax':[]} 
predict = {'pred_class': [], 'score': [], 'pred_xmin':[], 'pred_ymin':[], 'pred_xmax':[], 'pred_ymax':[], '~IoU': []} 


#Adds an entry of ground truth values to ground_truth
def add_to_dict(image):
    
    ground_truth["_image"].append(image)

    fullname = os.path.join(path_U, 'data/RGZdevkit2017/RGZ2017/Annotations', image + '.xml')
    tree = ET.parse(fullname)
    root = tree.getroot()

    #Get ground truth and add to table 
    for obj in root.findall('object'):
        ground_truth['act_class'].append(obj.find('name').text)  
        for box in obj.findall('bndbox'):
            ground_truth['act_xmin'].append(float(box.find('xmin').text) * (600.0/132.0) )
            ground_truth['act_xmax'].append(float(box.find('xmax').text) * (600.0/132.0) )    
            ground_truth['act_ymin'].append(float(box.find('ymin').text) * (600.0/132.0) )
            ground_truth['act_ymax'].append(float(box.find('ymax').text) * (600.0/132.0) ) 

#Writes a dictionary to a csv file
def dict_to_csv(dict):

    keys = sorted(dict.keys())
    print(dict) 
    with open("intern/" + subset + "_table.csv", "a+") as outfile:
        writer = csv.writer(outfile, delimiter = ",")
        #writer.writerow(keys)
        writer.writerows(zip(*[dict[key] for key in keys]))

## Computes the intersection over union given two bounding boxes 
## code sourced from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def vis_detections(im, class_name, dets,ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        # get a box with a highest score
        # try:
        #     max_score = np.max(dets[:, -1])
        #     inds = np.where(dets[:, -1] == max_score)[0][0:1]
        # except Exception as exp:
        #     print('inds == 0, but %s' % str(exp))
        return len(inds)
    #inds = range(dets.shape[0])
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=next(colors_), linewidth=1.0)
            )
        #cns = class_name.split('_')
        #class_name = '%sC%sP' % (cns[0], cns[1])
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.2f}'.format(class_name.replace('_', 'C_') + 'P', score),
                bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                fontsize=14, color='white')
                #bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                #fontsize=25, color='black')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    plt.axis('off')
    #plt.tight_layout()
    plt.draw()
    return len(inds)

def vis_detections_new(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -2] >= thresh)[0]
    if len(inds) == 0:
        thresh = np.max(dets[:, -2])
        inds = np.where(dets[:, -2] >= thresh)[0]

    for i in inds:
        
        bbox = dets[i, :4]
        score = dets[i, -2]
        

        if (class_name is None):
            class_name = CLASSES[int(dets[i, -1])]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=next(colors_), linewidth=1.5)
            )
        #cns = class_name.split('_')
        #class_name = '%sC%sP' % (cns[0], cns[1])
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.2f}'.format(class_name.replace('_', 'C_') + 'P', score),
                bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                fontsize=11, color='black')

        #ax.set_title(('{} detections with ' + 'p({} | box) >= {:.1f}').format(class_name, class_name,thresh),
         #       fontsize=14)
    
    #plt.tight_layout()
    #plt.draw()
    return len(inds)

def plt_detections(im, class_name, dets, ax, bboxs, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
    
        return len(inds)
    
    bbox = dets[inds[0], :4]
    score = dets[inds[0], -1]

    #Add ClaRAN's prediction to dictionary
    predict['pred_class'].append(class_name)
    predict['score'].append(score)
    predict['pred_xmin'].append(bbox[0])
    predict['pred_ymin'].append(bbox[1])
    predict['pred_xmax'].append(bbox[2])
    predict['pred_ymax'].append(bbox[3])
    
    #Draw ground truth boxes
    for i in range(len(ground_truth['act_xmin'])): 
        ax.add_patch(
            plt.Rectangle((ground_truth['act_xmin'][i], ground_truth['act_ymin'][i]),
                        ground_truth['act_xmax'][i] - ground_truth['act_xmin'][i],
                        ground_truth['act_ymax'][i] - ground_truth['act_ymin'][i], fill=False,
                        edgecolor='gray', linewidth=2.0, linestyle='--')
         )    

    #Draw proposed boxes
    for a_box in range(np.shape(bboxs)[0]): 

        bbox = bboxs[a_box, :4]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1], fill=False,
                        edgecolor=colors_2[a_box], linewidth=1.0)
            )
        
        #Adds IoU value to top left corner of each box 
        #gt = ground_truth['act_xmin'] + ground_truth['act_ymin'] + ground_truth['act_xmax'] + ground_truth['act_ymax']
        #ax.text(bbox[0], bbox[1], str(round(bb_iou(gt, bbox),2)), fontsize=5, color='black') 
    
    bbox = dets[inds[:], :4] 
    
    #ClaRAN bbox choice IoU
    #predict['~IoU'].append(bb_iou(gt, bbox)) 

    return len(inds)


def demo(sess, net, im_file, vis_file, fits_fn, 
         conf_thresh=0.8, eval_class=True, extra_vis_png=False, plot=False):
    """
    Detect object classes in an image using pre-computed object proposals.
    im_file:    The "fused" image file path
    vis_file:   The background image file on which detections are laid.
                Normallly, this is just the IR image file path
    fits_fn:    The FITS file path
    eval_class: True - use traditional per class-based evaluation style
                False - use per RoI-based evaluation

    """
    show_img_size = cfg.TEST.SCALES[0]

    if (not os.path.exists(im_file)):
        print('%s cannot be found' % (im_file))
        
        return -1
    
    # Add ground truth values to dictionary
    im = cv2.imread(im_file)
    im_file_name = im_file[5:26]
    add_to_dict(im_file_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    image_name = osp.basename(im_file)
    scores, boxes = im_detect(sess, net, im, save_vis_dir=None,
                             img_name=os.path.splitext(image_name)[0])
    boxes *= float(show_img_size) / float(im.shape[0])
    timer.toc()
    sys.stdout.write('Done in {:.3f} secs'.format(timer.total_time))
    sys.stdout.flush()

    im = cv2.imread(vis_file)

    my_dpi = 100
    fig = plt.figure()

    if plot: 
        fig.set_size_inches(show_img_size / my_dpi * 2, show_img_size / my_dpi)
        fig.suptitle(im_file_name, fontsize = 18)
        ax = fig.add_subplot(1, 2, 1) 
    else: 
        fig.set_size_inches(show_img_size / my_dpi, show_img_size / my_dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])

    im = cv2.resize(im, (show_img_size, show_img_size))
    im = im[:, :, (2, 1, 0)]

    ax.axis('off')
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_xlim([0, show_img_size])
    ax.set_ylim([show_img_size, 0])

    ax.imshow(im, aspect='equal')  

    if ((fits_fn is not None) and (not extra_vis_png)):
        patch_contour = fuse(fits_fn, im, None, sigma_level=4, mask_ir=False,
                             get_path_patch_only=True)
        ax.add_patch(patch_contour)
    NMS_THRESH = cfg.TEST.NMS #cfg.TEST.RPN_NMS_THRESH # 0.3

    tt_vis = 0
    bbox_img = []
    bscore_img = []
    num_sources = 0

    #if (eval_class):
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind : 4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]        

        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis]))#.astype(np.float32)

        #get copy of dets for plotting purposes
        bboxs = np.empty(np.shape(dets))
        np.copyto(bboxs, dets) 

        keep = nms(dets, NMS_THRESH)
        
        dets = dets[keep, :] 
        
        if plot: 
            num_sources += plt_detections(im, cls, dets, ax, bboxs, thresh=conf_thresh)
            fig.subplots_adjust(top=0.85)
        else: 
            num_sources += vis_detections(im, cls, dets, ax, thresh=conf_thresh)

        #add_to_csv(scores)
             


        #dets = np.hstack((dets, np.ones([dets.shape[0], 1]) * cls_ind))
        # if (dets.shape[0] > 0):
        #     bbox_img.append(dets)
        #     bscore_img.append(np.reshape(dets[:, -2], [-1, 1]))
    # else:
    #     for eoi_ind, eoi in enumerate(boxes):
    #         eoi_scores = scores[eoi_ind, 1:] # skip background
    #         cls_ind = np.argmax(eoi_scores) + 1 # add the background index back
    #         cls_boxes = boxes[eoi_ind, 4 * cls_ind : 4 * (cls_ind + 1)]
    #         cls_scores = scores[eoi_ind, cls_ind]
    #         dets = np.hstack((np.reshape(cls_boxes, [1, -1]),
    #                           np.reshape(cls_scores, [-1, 1])))#.astype(np.float32)
    #         dets = np.hstack((dets, np.ones([dets.shape[0], 1]) * cls_ind))
    #         bbox_img.append(dets)
    #         bscore_img.append(np.reshape(dets[:, -2], [-1, 1]))
    #
    # boxes_im = np.vstack(bbox_img)
    # scores_im = np.vstack(bscore_img)
    #
    # #if (not eval_class):
    # # a numpy float is a C double, so need to use float32
    # keep = nms(boxes_im[:, :-1].astype(np.float32), NMS_THRESH)
    # boxes_im = boxes_im[keep, :]
    # scores_im = scores_im[keep, :]
    #
    # keep_indices = range(boxes_im.shape[0])
    #num_sources = vis_detections(im, None, boxes_im[keep_indices, :], ax, thresh=conf_thresh)


    print(', found %d sources' % num_sources)

    #If no sources detected plot average position of detection boxes 
    if num_sources==0:
        boxes = np.reshape(boxes, (cfg.TEST.RPN_POST_NMS_TOP_N,7,4))
        bboxs = np.average(boxes,axis=1)
        #Draw ground truth boxes
        for i in range(len(ground_truth['act_xmin'])): 
            ax.add_patch(
                plt.Rectangle((ground_truth['act_xmin'][i], ground_truth['act_ymin'][i]),
                            ground_truth['act_xmax'][i] - ground_truth['act_xmin'][i],
                            ground_truth['act_ymax'][i] - ground_truth['act_ymin'][i], fill=False,
                            edgecolor='gray', linewidth=2.0, linestyle='--')
             ) 
        #Draw proposed boxes
        for a_box in range(np.shape(bboxs)[0]): 

            bbox = bboxs[a_box, :4]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor=colors_2[a_box], linewidth=1.0)
                )

    ##Plot scores for each box as a grouped histogram
    if plot: 
        axa = fig.add_subplot(1, 2, 2)
        barwidth = (1-0.1)/cfg.TEST.RPN_POST_NMS_TOP_N 
        axa.set_ylim([0,1])
        r1 = np.arange(len(scores[0,:]))

        for i in range(0,int(np.shape(scores)[0])):
            
            axa.bar(r1, scores[i, :], color = colors_2[i], width = barwidth, label='det_' + str(i+1))
            r1 = [x + barwidth for x in r1]

        axa.set_xticks([r + 0.45 for r in range(len(scores[0,:]))])
        axa.set_xticklabels(CLASSES[:], fontsize = 8)

        #Draw horizontal line at threshold, add legend and title
        axa.axhline(conf_thresh, linestyle='--', color='black', linewidth=1.0)
        plt.legend(loc = 'upper right', fontsize='x-small')
        axa.set_title('Class Scores for each Detection')

        # Generate text string for ClaRAN's prediction
        claran_text = '' 
        for p in range(len(predict['pred_class'])):
            claran_text += str(predict['pred_class'][p].replace('_', 'C_') + 'P ' + "{:.2f}".format(predict['score'][p])) + " "
        ax.text(5, 20, '{:s}'.format('ClaRAN: '+ claran_text),
                bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                fontsize=8, color='black')

        # Generate text string for ground truth
        gt_text = '' 
        for q in ground_truth['act_class']:
            gt_text += q.replace('_', 'C_') + 'P '
        ax.text(5,40, '{:s}'.format('Ground Truth: ' + gt_text),
                bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                fontsize=8, color='black')

        plt.tight_layout() 
        fig.subplots_adjust(top=0.85)
    
    plt.draw() 

    # save results to CSV if ClaRAN has found a single source
    #if num_sources == 1: 
        
        #dict_to_csv( dict(ground_truth.items() + predict.items()) )

    return 0

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Run a demo detector')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use for GPUs',
                        default=0, type=int)
    parser.add_argument('--radio', dest='radio_fits',
                        help='full path of the radio fits file (compulsory)',
                        default=None, type=str)
    parser.add_argument('--ir', dest='ir_png',
                        help='full path of the infrared png file (compulsory)',
                        default=None, type=str)
    parser.add_argument('--output', dest='fig_path', help='Output path for detections',
                        default='.')
    parser.add_argument('--p-value', dest='conf_thresh', help='P-value',
                        default=0.8, type=float)
    parser.add_argument('--evaleoi', dest='eval_eoi',
                        help='evaluation based on EoI',
                        action='store_true', default=False)
    parser.add_argument('--model', dest='model', help='which pre-trained model to load',
                        default='D4')
    parser.add_argument('--radio-png', dest='radio_png',
                        help='full path of the radio png file (only for D1 method)',
                        default=None, type=str)
    parser.add_argument('--vis-png', dest='vis_png',
                        help='full path of the visualisation png file',
                        default=None, type=str)
    parser.add_argument('--plot', dest='plot', 
                        help='plot bounding boxes and scores', action='store_true', default=False)

    args = parser.parse_args()
    if ('D1' != args.model):
        if (args.radio_fits is None or args.ir_png is None):
            parser.print_help()
            sys.exit(1)

        if (not osp.exists(args.radio_fits)):
            print('Radio fits %s not found' % args.radio_fits)
            sys.exit(1)

        if (not osp.exists(args.ir_png)):
            print('Infrared png %s not found' % args.ir_png)
            sys.exit(1)

        if (not args.model in ['D4', 'D5', 'D1', 'D3']):
            print('Unknown model: %s' % args.model)
            sys.exit(1)
    else:
        if (args.radio_png is None):
            print('D1 method must specify radio png path with option "--radio-png"')
            sys.exit(1)
        elif (not osp.exists(args.radio_png)):
            print('Radio png not found: %s' % args.radio_png)
            sys.exit(1)

    if args.device.lower() == 'gpu':
        cfg.USE_GPU_NMS = True
        cfg.GPU_ID = args.device_id
        device_name = '/{}:{:d}'.format(args.device, args.device_id)
        print(device_name)
    else:
        cfg.USE_GPU_NMS = False

    return args

def hard_code_cfg():
    """
    This could be parameters potentially. Hardcoded for now
    """
    cfg.TEST.HAS_RPN = True
    cfg.TEST.RPN_MIN_SIZE = 4
    cfg.TEST.RPN_POST_NMS_TOP_N = 5
    cfg.TEST.NMS = 0.3
    cfg.TEST.RPN_NMS_THRESH = 0.5
    cfg.TEST.SCALES = (600,)
    cfg.TEST.MAX_SIZE = 2000

def fuse_radio_ir_4_pred(radio_fn, ir_fn, out_dir='/tmp', model='D4'):
    """
    return the full path to the generated fused file
    """
    if (model != 'D5'):
        nsz = None
        if ('D3' == model):
            mask_ir = False
        else:
            mask_ir = True
    else:
        nsz = cfg.TEST.SCALES[0] #i.e. 600
        mask_ir = True
    return fuse(radio_fn, ir_fn, out_dir, new_size=nsz, mask_ir=mask_ir)    


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #hide tensorflow warnings
    args = parse_args()
    if ('D1' == args.model):
        im_file = args.radio_png
        vis_file = args.radio_png
    else:
        im_file = fuse_radio_ir_4_pred(args.radio_fits, args.ir_png, model=args.model)
        vis_file = args.ir_png if args.vis_png is None else args.vis_png
    #print("im_file", im_file)
    if (im_file is None):
        print("Error in generating contours")
        sys.exit(1)
    hard_code_cfg()
    net = get_network('rgz_test')
    iter_num = 80000
    if ('D3' == args.model):
        iter_num = 60000
    model_weight = osp.join(get_rgz_root(), 'data/pretrained_model/rgz/%s/VGGnet_fast_rcnn-%d' % (args.model, iter_num))
    if (not osp.exists(model_weight + '.index')):
        print("Fail to load rgz model, have you done \'python download_data.py\' already?")
        sys.exit(1)

    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sys.stdout.write('Loading RGZ model from {:s}... '.format(model_weight))
    sys.stdout.flush()
    stt = time.time()
    saver.restore(sess, model_weight)
    print("Done in %.3f seconds" % (time.time() - stt))
    sys.stdout.write("Detecting radio sources... ")
    sys.stdout.flush()
    ret = demo(sess, net, im_file, vis_file, args.radio_fits, conf_thresh=args.conf_thresh,
               eval_class=(not args.eval_eoi), extra_vis_png=args.vis_png is not None, plot=args.plot)

    if (-1 == ret):
        print('Fail to detect in %s' % args.radio_fits)
    else:
        im_name = osp.basename(im_file)
        if args.plot: end='_plot.png' 
        else: end='_pred.png' 
        output_fn = osp.join(args.fig_path, im_name.replace('.png', end))
        plt.savefig(output_fn, dpi=150)
        plt.close()
        print('Detection saved to %s' % output_fn)
        if ('D1' != args.model):
            for trf in [im_file]:
                if (osp.exists(trf)):
                    try:
                        os.remove(trf)
                    except:
                        pass
