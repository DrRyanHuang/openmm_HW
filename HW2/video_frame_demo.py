# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
import sys; sys.path.append(".")
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import cv2
import numpy as np
import os

'''
"test_video.mp4",
"mask_rcnn_r50_fpn_1x_coco.py",
"epoch_199.pth",
"--out-file-dir", "out_dir"
'''

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('video', help='video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')# epoch_199.pth
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument('--out-file-dir', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    
    cap = cv2.VideoCapture(args.video)
    os.makedirs(args.out_file_dir, exist_ok=True)
    im_id = 0
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        result = inference_detector(model, frame)
        
        det_res = result[0][0]
        seg_res = np.array(result[1][0])
        
        remain = det_res[:, -1] > args.score_thr
        colored = seg_res[remain].sum(axis=0) > 0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray,  cv2.COLOR_GRAY2BGR)
        
        res = gray.copy()
        res[colored] = frame[colored]
        
        save_name = "%s.png" % im_id
        save_path = os.path.join(args.out_file_dir, save_name)
        im_id += 1
        
        cv2.imwrite(save_name, res)
        
    # # show the results
    # show_result_pyplot(
    #     model,
    #     args.img,
    #     result,
    #     palette=args.palette,
    #     score_thr=args.score_thr,
    #     out_file=args.out_file)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
