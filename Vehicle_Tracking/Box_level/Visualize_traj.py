from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import imutils
import argparse
import os
from tqdm import tqdm,trange
import numpy as np
import joblib
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
def main():
  interval=30
  npy_dir='./Ori_track/'
  video_dir='./Raw_video/'
  video_set=os.listdir(video_dir)
  #video_set=['23.mp4']
  store_dir='./Trajectory_V3/'
  if not os.path.exists(store_dir):
      os.makedirs(store_dir) 

  for source in video_set:

    video_name=video_dir+source
    result_dir=store_dir+source+'/'

    if not os.path.exists(result_dir):
      os.makedirs(result_dir) 
    img_dir=result_dir+'img/'
    traj_dir =result_dir+'traj/'
    if not os.path.exists(img_dir):
      os.makedirs(img_dir)
    if not os.path.exists(traj_dir):
      os.makedirs(traj_dir)

    name=source.split('.')
    npy_file = npy_dir+name[0]+'.npy'
    frame_result_list=np.load(npy_file,allow_pickle=True)

    cap = cv2.VideoCapture(video_name)
    frame_all=int(cap.get(7))
    np.random.seed(2021)
    color = np.random.randint(0,255,(600,3))

    time_stamp=range(0,frame_all,interval)
    
    for i in tqdm(time_stamp):
      
      id_dict={}
      first=i
      second=min(i+interval,frame_all-1)
      cap.set(cv2.CAP_PROP_POS_FRAMES,first)
      _, img_first = cap.read()
      cap.set(cv2.CAP_PROP_POS_FRAMES,second)
      _, img_second = cap.read()

      mixed_img = cv2.addWeighted(img_first, 0.5, img_second, 0.5, 0)
      new_list=np.array(frame_result_list)
      first_draw=True
      for boxes in new_list[first:second+1]:
        if len(boxes):
          if first_draw:
            mixed_img=plot_bboxes(mixed_img, boxes)
            first_draw=False
          for index,box in enumerate(boxes):

            a,b=int((box[0]+box[2])/2),int((box[1]+box[3])/2)
            car_id=int(box[5])
            if car_id not in id_dict.keys():
              id_dict[car_id]=[]
            id_dict[car_id].append([a,b])
            #mixed_img=cv2.putText(mixed_img, str(car_id), (a,b), cv2.FONT_HERSHEY_COMPLEX, 1, color[car_id].tolist(), 1)
            mixed_img = cv2.circle(mixed_img,(a,b),3,color[car_id].tolist(),-1)

      cv2.imwrite(img_dir+str(first)+'_'+str(second)+'.jpg',mixed_img)
      joblib.dump(id_dict,traj_dir+str(first)+'_'+str(second)+'.pkl')

    cap.release()        



if __name__ == '__main__':

    main()