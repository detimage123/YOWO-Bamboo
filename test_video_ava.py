import argparse
import cv2
import os
import time
import numpy as np
import torch
from PIL import Image

from dataset.transforms import BaseTransform
from utils.misc import load_weight
from config import build_dataset_config, build_model_config
from models.detector import build_model



def parse_args():
    parser = argparse.ArgumentParser(description='YOWO')

    # basic
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--vis_thresh', default=0.15, type=float,
                        help='threshold for visualization')
    parser.add_argument('--video', default=NOne, type=str,
                        help='AVA video name.')
    parser.add_argument('-d', '--dataset', default='ava_v2.2',
                        help='ava_v2.2')

    # model
    parser.add_argument('-v', '--version', default='yowo', type=str,
                        help='build YOWO')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')

    return parser.parse_args()
                    

@torch.no_grad()
def run(args, d_cfg, model, device, transform, class_names):
    # path to save
    video_name_index = args.video.rfind('/')
    video_name = args.video[video_name_index + 1:-4]

    save_path = os.path.join(args.save_folder, 'ava_video', video_name)
    os.makedirs(save_path, exist_ok=True)

    # path to video
    # path_to_video = os.path.join(d_cfg['data_root'], 'videos_15min', args.video)
    path_to_video = os.path.join(args.video)

    # video
    video = cv2.VideoCapture(path_to_video)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_size = (frame_width, frame_height)
    save_name = os.path.join(save_path, 'detection.mp4')
    fps = video.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(save_name, fourcc, fps, save_size)
    print(f'视频{video_name} 分辨率：{frame_width}x{frame_height} fps:{fps} 时长：{total_frames / fps}s')
    t0 = time.time()
    video_clip = []
    while(True):
        ret, frame = video.read()
        
        if ret:
            # to PIL image
            frame_pil = Image.fromarray(frame.astype(np.uint8))

            # prepare
            if len(video_clip) <= 0:
                for _ in range(d_cfg['len_clip']):
                    video_clip.append(frame_pil)

            video_clip.append(frame_pil)
            del video_clip[0]

            # orig size
            orig_h, orig_w = frame.shape[:2]

            # transform
            x, _ = transform(video_clip)
            # List [T, 3, H, W] -> [3, T, H, W]
            x = torch.stack(x, dim=1)
            x = x.unsqueeze(0).to(device) # [B, 3, T, H, W], B=1

            t0 = time.time()
            # inference
            batch_bboxes = model(x)
            print("inference time ", time.time() - t0, "s")

            # batch size = 1
            bboxes = batch_bboxes[0]

            # visualize detection results
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox[:4]
                det_conf = float(bbox[4])
                cls_out = [det_conf * cls_conf for cls_conf in bbox[5:]]
            
                # rescale bbox
                x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
                y1, y2 = int(y1 * orig_h), int(y2 * orig_h)

                cls_scores = np.array(cls_out)
                indices = np.where(cls_scores > 0.15)
                scores = cls_scores[indices]
                indices = list(indices[0])
                scores = list(scores)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if len(scores) > 0:
                    blk   = np.zeros(frame.shape, np.uint8)
                    font  = cv2.FONT_HERSHEY_SIMPLEX
                    coord = []
                    text  = []
                    text_size = []

                    for _, cls_ind in enumerate(indices):
                        text.append("[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind]))

                        # 增大文字大小
                        fontScale = 0.9
                        thickness = 2
                        line_space=20
                        text_size = cv2.getTextSize(text[-1], font, fontScale=fontScale, thickness=thickness)[0]

                        # 调整垂直位置，确保有足够的空间
                        coord.append((x1 + 3, y1 - 7 + 10 * _ + (len(text) - 1) * line_space))  # 增加垂直方向的偏移量

                        # 绘制背景矩形，确保足够的空间
                        cv2.rectangle(blk,
                                      (coord[-1][0] - 1, coord[-1][1] - text_size[1] - 7),  # 垂直方向增加空间
                                      (coord[-1][0] + text_size[0] + 1, coord[-1][1] + 2),  # 顶部减少一些空间，底部保持
                                      (0, 255, 0), cv2.FILLED)

                    frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)

                    for t in range(len(text)):
                        # 使用更大的fontScale绘制文字
                        cv2.putText(frame, text[t], coord[t], font, fontScale, (0, 0, 0), thickness)

                    # for _, cls_ind in enumerate(indices):
                    #     text.append("[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind]))
                    #     # 增大文字大小
                    #     text_size = cv2.getTextSize(text[-1], font, fontScale=0.6, thickness=2)[0]  # 例如，从0.25增加到0.5
                    #     coord.append((x1 + 3, y1 + 7 + 10 * _))  # 这里你可能还需要根据新的文字大小调整垂直位置
                    #
                    #     # 绘制背景矩形，可能需要根据新的文字大小调整位置和大小
                    #     cv2.rectangle(blk, (coord[-1][0] - 1, coord[-1][1] - text_size[1] + 6),
                    #                   # 增加了-text_size[1]-2来确保足够的垂直空间
                    #                   (coord[-1][0] + text_size[0] + 1, coord[-1][1] + 1),  # 减少了垂直大小，但你可以根据需要调整
                    #                   (0, 255, 0), cv2.FILLED)
                    #
                    # frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)
                    #
                    # for t in range(len(text)):
                    #     # 如果你想让文字更清晰，可以在这里也增加fontScale
                    #     cv2.putText(frame, text[t], coord[t], font, 0.5, (0, 0, 0), 1)  # 同样从0.25增加到0.5


            # save
            out.write(frame)

            if args.show:
                # show
                cv2.imshow('key-frame detection', frame)
                cv2.waitKey(1)

        else:
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    class_names = d_cfg['label_map']
    num_classes = 80

    # transform
    basetransform = BaseTransform(
        img_size=d_cfg['test_size'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std']
        )

    # build model
    model = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False
        )

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval
    model = model.to(device).eval()

    # run
    run(args=args, d_cfg=d_cfg, model=model, device=device,
        transform=basetransform, class_names=class_names)
