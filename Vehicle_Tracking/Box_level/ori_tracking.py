from AIDetector_pytorch import Detector
import imutils
import cv2
import argparse
import os
from tqdm import tqdm,trange
def main(opt):

    func_status = {}
    func_status['headpose'] = None
    #portion = os.path.splitext(opt.video_path)
    video_name='/content/drive/MyDrive/Colab Notebooks/aic21-track4-test-data/'+opt.video+'.mp4'
    det = Detector('./txt_dir/'+opt.video+'.txt')
    #print(det.weights)
    cap = cv2.VideoCapture(video_name)


    for i in trange(int(cap.get(7))):
        _, im = cap.read()
        if im is None:
            break
        
        result = det.feedCap(im, func_status)


    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='1', help='input video')
    
    opt = parser.parse_args()
    main(opt)