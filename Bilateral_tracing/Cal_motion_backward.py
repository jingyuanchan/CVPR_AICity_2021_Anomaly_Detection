import numpy as np
import cv2
import joblib
#from google.colab.patches import cv2_imshow
all_result=joblib.load('./final_fuse.pkl')
import numpy as np
from sklearn.neighbors import NearestNeighbors
np.random.seed(2020)
from collections import Counter

for video_num in all_result.keys():
    videoWriter=None
    cap = cv2.VideoCapture('./aic21-track4-test-data/'+str(video_num)+'.mp4')


    #rint(all_result)
    start,box=int((all_result[video_num][0])*30),all_result[video_num][1]
    #rint(start,box)
    from shapely import geometry

    def if_inPoly(polygon, Points):
        line = geometry.LineString(polygon)
        point = geometry.Point(Points)
        polygon = geometry.Polygon(line)
        return polygon.contains(point)


    # ShiTomasi 
    feature_params = dict( maxCorners = 1000,
                           qualityLevel = 0.05,
                           minDistance = 5,
                           blockSize = 3 )

    # lucas kanade
    lk_params = dict( winSize  = (9,9),
                      maxLevel = 3,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    color = np.random.randint(0,255,(300,3))

 
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    ret, old_frame = cap.read()

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)


    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    pre=p0.reshape(-1,2)

    dst=[]
    dst_id_list=[]


    for i in pre:
      point=(i[0],i[1])
      poly=[(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])]
      line = geometry.LineString(poly)
      point = geometry.Point(point)
      polygon = geometry.Polygon(line)
      if polygon.contains(point):
        dst.append(i)
        dst_id_list.append(10)
        break

    all_supple=None
    all_supple_id_list=None


    pad=0.2
    x_min=abs(box[0]-box[2])*pad+box[0]
    y_min=abs(box[1]-box[3])*pad+box[1]
    x_max=box[2]-abs(box[0]-box[2])*pad
    y_max=box[3]-abs(box[1]-box[3])*pad
    p_num=50
    x = np.random.randint(x_min,x_max,size=p_num)
    y= np.random.randint(y_min,y_max,size=p_num)
    supple=[np.array([a,b],dtype=np.float32) for a,b in zip(x,y)]
    supple_id_list=[10]*len(supple)
    if all_supple==None and all_supple_id_list==None:
      all_supple=supple
      all_supple_id_list=supple_id_list
    else:
      all_supple+=supple
      all_supple_id_list+=supple_id_list

    p0=np.array(dst+all_supple).reshape(-1,1,2)
    p0_id_list=np.array(dst_id_list+all_supple_id_list)

    mask = np.zeros_like(old_frame)
    ellapse=(13)*30
    all_frame=int(cap.get(7))
    
    for i in range(ellapse):
        cap.set(cv2.CAP_PROP_POS_FRAMES,start-i)
        ret,frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # 选取好的跟踪点
        if np.shape(p1)!=():
            good_new = p1.reshape(-1,2)
            good_old = p0.reshape(-1,2)
            
            if good_new.shape[0]>6:

                neigh = NearestNeighbors(n_neighbors=6, radius=1)
                neigh.fit(good_new)

                bn_list=[neigh.kneighbors([new], 6, return_distance=False)[0] for new in good_new]
                #print(bn_list)
                I_density_list=[]
                for i in bn_list:
                    #import pdb; pdb.set_trace()
                    i_cor=[good_new[int(index)] for index in i]
                    I_density=np.sum([np.log(6/np.sum(neigh.kneighbors([new], 6, return_distance=True)[1])) for new in i_cor])/6
                    I_density_list.append(I_density)

                density_list=[6/np.sum(neigh.kneighbors([new], 6, return_distance=True)[1]) for new in good_new]

                for i,(density,I_density) in enumerate(zip(density_list,I_density_list)):

                    a=np.sign(-1)*np.log(density)-I_density
                    #print(a)
                    if a>6.6:
                        #import pdb; pdb.set_trace()
                        st[i]=0
                    x1,y1=good_new[i].ravel()
                    x2,y2=good_old[i].ravel()
                    dis=((x1-x2)**2+(y1-y2)**2)**0.5
                    #print(dis)
                    if dis>20:
                        st[i]=0
                
            if good_new.shape[0]<=6:
                print("point less than 6")
                break
            #print(st)

            if np.shape(p1)!=():
                good_new = p1[st==1]
                good_old = p0[st==1]

                for i,(new,old) in enumerate(zip(good_new,good_old)):

                    a,b = new.ravel()
                    c,d = old.ravel()

                    car_id=int(p0_id_list[i])
                    mask = cv2.line(mask, (a,b),(c,d), color[car_id].tolist(),2)

                img = cv2.add(frame,mask)

#                 cv2.imshow('frame',img)
#                 k = cv2.waitKey(1) & 0xff
#                 if k == 27:
#                     break


                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)
                
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter('./back_traj/'+str(video_num)+'.mp4', fourcc, 30, (img.shape[1], img.shape[0]))

        videoWriter.write(img)

    #cv2.destroyAllWindows()
    cap.release()
    videoWriter.release()