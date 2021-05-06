

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import imutils
import argparse
import os
from tqdm import tqdm,trange
import numpy as np
def main(opt):
  test_path='./Forward_track_V2/'
  video_path='./Forward_video_V2/'
  if not os.path.exists(test_path):
       
      os.makedirs(test_path) 
  if not os.path.exists(video_path):
       
      os.makedirs(video_path) 
  for source in range(int(opt.start), int(opt.end)+1):

    video_name='/content/drive/MyDrive/Colab Notebooks/yolov5/ForwardBg-Frame/'+str(source)+'.mp4'
    npy_file = '/content/drive/MyDrive/Colab Notebooks/New_yolov5/yolov5/Forward_second_weight2/'+str(source)+'.npy'
    npy_path =test_path+str(source)+'.npy'
    cap = cv2.VideoCapture(video_name)
    frame_all=int(cap.get(7))
    b_list=np.load(npy_file,allow_pickle=True)
    #display(b_list)
    videoWriter = None

    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    cfg = get_config()
    cfg.merge_from_file("deep_sort/configs/deep_sort_ori.yaml")
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                          max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=0.4,
                          nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                          max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                          use_cuda=True)

    def plot_bboxes(image, bboxes, line_thickness=1):
        # Plots one bounding box on image img
        tl = line_thickness or round(
            0.001 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, (0,255,0), thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, (0,255,0), -1, cv2.LINE_AA)  # filled
            cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        return image

    def update_tracker(bboxes, image):
            bbox_xywh = []
            confs = []
            bboxes2draw = []
            
            if len(bboxes):

                # Adapt detections to deep sort input format
                for box,_, conf in bboxes:
                    
                    #obj = [(box[0])/800,(box[1]-box[3])/410,(box[0]+box[2])/800,(box[1])/410
                    obj  =[int(box[0])+int(box[2])/2, int(box[1])-int(box[3])/2,
                        int(box[2]), int(box[3])
                          ]
                
                    bbox_xywh.append(obj)
                    confs.append(conf)

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, image)

                for value in list(outputs):
                    x1,y1,x2,y2,track_id = value
                    bboxes2draw.append(
                        (int(x1), int(y1), int(x2), int(y2), '', track_id)
                    )

            image = plot_bboxes(image, bboxes2draw)
            

            return image,bboxes2draw


    all_box=[]
    for i in tqdm(enumerate(range(frame_all))):
      _, im = cap.read()
      if im is None:
        break

      result,frame_box=update_tracker(b_list[i[0]],im)
      all_box.append(frame_box)
      if videoWriter is None:
              fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # opencv3.0
              videoWriter = cv2.VideoWriter(video_path+str(source)+'.mp4', fourcc, 30, (im.shape[1], im.shape[0]))

      videoWriter.write(result)
    np.save(npy_path,all_box)

    cap.release()        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='1', help='start')  # file/folder, 0 for webcam
    parser.add_argument('--end', type=str, default='1', help='start')  # file/folder, 0 for webcam
    
    opt = parser.parse_args()
    main(opt)