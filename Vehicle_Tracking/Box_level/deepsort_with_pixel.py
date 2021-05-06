
import numpy as np
import os
def compute_iou(rec1, rec2):
 
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0
def main():
  base_dir='./Forward_track/'
  file_list=os.listdir(base_dir)
  video_list=[7,9,11,23,26,33,34,35,36,40,41,43,44,45,48,49,50,67,70,72,73,74,80,86,89,91,92,99,104,105,109,119,120,124,132,133,134,137,139,144,146,148]
  #video_list=[7]
  result2store=[]
  for source in video_list:
    all_result=np.load(base_dir+str(source)+'.npy',allow_pickle=True)

    all_cars_tracked=[]
    for frame_result in all_result:
      if len(frame_result):
        for i in frame_result:
          all_cars_tracked.append(i[5])
    #统计id的出现次数，减少检测目标
    from collections import Counter
    count_id=dict(Counter(all_cars_tracked).most_common(20))



    new_id={}

    #给每个id创建dict储存帧数和相应bounding box
    for frame_result in enumerate(all_result):
      if len(frame_result[1]):
        for box in frame_result[1]:
          if box[5] in count_id.keys():
            if box[5]not in new_id.keys():
              new_id[box[5]]={'frame':[],'box':[],'center':[]}
            new_id[box[5]]['frame'].append(frame_result[0])
            new_id[box[5]]['box'].append([box[0],box[1],box[2],box[3]])
            new_id[box[5]]['center'].append([(box[0]+box[2])/2,(box[1]+box[3])/2])

    combine_id={}
    scaned=[]
    for key in new_id.keys():
      if key in scaned:
        continue
      time_stamp=new_id[key]['frame']
      boxes=new_id[key]['box']
      m1=np.mean(boxes,axis=0)

      for key2 in new_id.keys():
        if key2 in scaned:
          continue
        boxes2=new_id[key2]['box']
        m2=np.mean(boxes2,axis=0)
        iou=compute_iou(m1,m2)

        if iou>0.4:
          if key==key2:
            combine_id[key]={'frame':new_id[key]['frame'],'box':new_id[key]['box'],'center':new_id[key]['center']}
            scaned.append(key2)
            scaned.append(key)
          if key!=key2:
            

            combine_id[key]['frame']+=new_id[key2]['frame']
            combine_id[key]['box']+=new_id[key2]['box']
            combine_id[key]['center']+=new_id[key2]['center']
            scaned.append(key2)
            scaned.append(key)
      
    mid_id={}

    #开始后处理，eliminate unstable tracking result base on the time
    for key in combine_id.keys():
      time_stamp=combine_id[key]['frame']
      boxes=combine_id[key]['box']
      centers=np.array(combine_id[key]['center'])
      m=np.std(centers,axis=0)


      for first,box_first in zip(time_stamp,boxes): 
      #adopt the id that last for more than certain time
        time=1
        if len(time_stamp)>200:
          #devide the time in to five region,ten sec per region
          for gap in range(5):
            frame_num_list=[]
            min_frame=gap*300+first
            max_frame=(gap+1)*300+first
            
            for frame in time_stamp:
              current=int(frame)
              if current>=min_frame and current<max_frame:
                frame_num_list.append(current)
            #define a stable tracking result if it appears for a cretain times in a region
            if len(frame_num_list)>10:
              time+=1
          #a stable result must be counted four time out of five
          if time>2:
    
                    #print(key,first/30,box_first)
                    result2store.append([source,first/30,box_first[0],box_first[1],box_first[2],box_first[3]])
                    break
  np.save('forward_middle.npy',result2store)
  dict_pixel={}
  all_result=np.load('fuse_pre.npy',allow_pickle=True).reshape(35,6)


  for i in all_result:
    #print(i[0])
    k=i
    if k[0] not in dict_pixel.keys():
      dict_pixel[k[0]]=[]
    dict_pixel[k[0]].append([k[1],k[2],k[3],k[4],k[5]])

    dict_box={}
  all_result2=np.load('forward_middle.npy',allow_pickle=True)


  for i in all_result2:
    #print(i[0])
    k=i
    if k[0] not in dict_box.keys():
      dict_box[k[0]]=[]
    
    dict_box[k[0]].append([k[1],k[2],k[3],k[4],k[5]])

    #print('Video_num:{0} Start:{1:.4f} Box:{2}'.format(k[0],k[1],[k[2],k[3],k[4],k[5]]))


  new_box={}
  for b_key in dict_box.keys():
    if b_key not in new_box.keys():
      new_box[b_key]=[]
    if b_key not in dict_pixel.keys():
      if int(b_key)==86:
        new_box[b_key].append(dict_box[b_key][1])
      if int(b_key)==144:
        new_box[b_key].append(dict_box[b_key][-2])
      else:
        new_box[b_key].append(dict_box[b_key][0])
    elif b_key in dict_pixel.keys():
      iou_list=[]
      for i in dict_box[b_key]:
        result_deepsort=i
        result_pixel=dict_pixel[b_key][0]
        box_deepsort=[result_deepsort[1],result_deepsort[2],result_deepsort[3],result_deepsort[4]]
        box_pixel=[result_pixel[1],result_pixel[2],result_pixel[3],result_pixel[4]]
        iou=compute_iou(box_deepsort,box_pixel)

        iou_list.append(iou)

      b_array=1-(abs(np.array(dict_box[b_key])[:,0]-dict_pixel[b_key][0][0])/1)
      iou_list=np.array(iou_list)
      
      mix_list=b_array*0.6+iou_list*0.4
      index=np.argmax(b_array)

      #print("Video Num:{0}  IoU:{1:.5f}".format(b_key,iou_list[index]))
    
      new_box[b_key].append(dict_box[b_key][index])
    #print(new_box)
  final_result=[]
  with open('forward_with_pixel.txt',"w") as track: 

    for i in new_box.keys():
      
      k=new_box[i][0]
      final_result.append([i,k[0],k[1],k[2],k[3],k[4]])
      instr=str(i)+' '+str(k[0])+' \n'
      track.write(instr)
      

      #print(k[0])

  np.save('forward_with_pixel.npy',final_result)
    #print('Video_num:{0} Start:{1:.4f} Box:{2}'.format(i,k[0],[k[1],k[2],k[3],k[4]]))


    # print(result2store)

if __name__ == '__main__':
  main()