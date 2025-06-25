import os
import cv2 as cv
import pandas as pd
import numpy as np
import sys
from utils_aict22 import *



def main():             
    video_ids_file='/mnt/works/ActionRecognition/AI_CITY_2022/code/data/A2_224x224/video_ids.csv'
    video_ids_data = pd.read_csv(video_ids_file)
    video_ids=list(video_ids_data.values) 

    testSet='A2'     
    output_dir='./output/Rightside_window/%s' % (testSet)


    with open(os.path.join(output_dir,'submit.txt'),'w') as f_submit:
        for i in range (len(video_ids)):
            # video_id,dashboard,read_view,right_side=video_ids[i]
            video_id,_,_,video=video_ids[i]
            output_file='%s/tmp3_%s.csv' % (output_dir,video)
            if not os.path.exists(output_file):
                print(output_file, ' not found')
                break
            output_data = pd.read_csv(output_file)
            output_data=list(output_data.values)
            for i in range (len(output_data)):
                _, cls1, start_time, end_time=output_data[i]
                if cls1 !=18:
                    str="%s %02d %03d %03d\n" % (video_id, cls1, start_time, end_time)
                    f_submit.write(str)        

if __name__ == '__main__':
    main()