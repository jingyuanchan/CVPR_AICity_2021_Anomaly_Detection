import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnns
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utilsv2.general import xyxyn2xlylwh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from tqdm import tqdm

def detect(save_img=False):
    start,end, imgsz = opt.start, opt.end, opt.img_size

    test_path='./Forward_all_video_second_weight2/'
    test_npy_path='./Forward_second_weight2/'

    if not os.path.exists(test_path):
       
        os.makedirs(test_path) 
    if not os.path.exists(test_npy_path):
       
        os.makedirs(test_npy_path) 
  
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load('/content/drive/MyDrive/Colab Notebooks/yolov5/weights/new.pt', map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    for source in range(int(start), int(end)+1):
      vid_path, vid_writer = None, None
      save_path=test_path+str(source)+'.mp4'
      save_img = True
      dataset = LoadImages('/content/drive/MyDrive/Colab Notebooks/yolov5/ForwardBg-Frame/'+str(source)+'.mp4', img_size=imgsz, stride=stride)

      # Get names and colors
      names = model.module.names if hasattr(model, 'module') else model.names
      colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

      # Run inference
      if device.type != 'cpu':
          model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
      t0 = time.time()
      all_npy=[]
      for path, img, im0s, vid_cap in tqdm(dataset):
          img = torch.from_numpy(img).to(device)
          img = img.half() if half else img.float()  # uint8 to fp16/32
          img /= 255.0  # 0 - 255 to 0.0 - 1.0
          if img.ndimension() == 3:
              img = img.unsqueeze(0)

          # Inference
          t1 = time_synchronized()
          pred = model(img, augment=opt.augment)[0]

          # Apply NMS
          pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
          t2 = time_synchronized()

          # Apply Classifier
          if classify:
              pred = apply_classifier(pred, modelc, img, im0s)

          # Process detections
          npy_res=[]
          for i, det in enumerate(pred):  # detections per image
        
              p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

              gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
              if len(det):
                  # Rescale boxes from img_size to im0 size
                  det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                  # Print results
                  for c in det[:, -1].unique():
                      n = (det[:, -1] == c).sum()  # detections per class
                      #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                  # Write results

                  for id_car,k in enumerate(reversed(det)):
                      *xyxy, conf, cls=k
                      if save_img or view_img:
                        xywh = (xyxyn2xlylwh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        npy_res.append([[*xywh],id_car,float(conf)]) 
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
              # Print time (inference + NMS)
            

              
              if save_img:
                      if vid_path != save_path:  # new video
                          vid_path = save_path
                          if isinstance(vid_writer, cv2.VideoWriter):
                              vid_writer.release()  # release previous video writer

                          fourcc = 'mp4v'  # output video codec
                          fps = vid_cap.get(cv2.CAP_PROP_FPS)
                          w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                          h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                          vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                      vid_writer.write(im0)

          all_npy.append(npy_res)
      #print(all_npy)
      np.save(test_npy_path+str(source)+'.npy',all_npy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='1', help='start')  # file/folder, 0 for webcam
    parser.add_argument('--end', type=str, default='1', help='start')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
