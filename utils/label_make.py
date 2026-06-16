import os
import cv2
from tqdm import tqdm

def crowd_convert():
    root = r"F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\object-detection-crowdai"
    labels = os.path.join(root, 'labels.csv')
    fr = open(labels, 'r', newline='')
    labels_dict = {}
    newlines = []
    for i,line in enumerate(fr):
        if i==0:continue
        info = line.strip().split(',')
        name = root + '/' + info[4]
        x1 = info[0]
        y1 = info[1]
        x2 = info[2]
        y2 = info[3]
        cls = 0
        if info[5] == 'Car':cls = 0
        elif info[5] == 'Truck':cls = 0
        elif info[5] == 'Pedestrian':cls = 1
        if name in labels_dict:
            labels_dict[name].append(','.join([x1,y1,x2,y2,str(cls)]))
        else:
            labels_dict[name] = [name, ','.join([x1,y1,x2,y2,str(cls)])]
    for lab in labels_dict:
        newline = ' '.join(labels_dict[lab])+'\n'
        newlines.append(newline)

    fw = open(r'F:\1_project_store\2_dev_info\3_data_set\1_cv\7_car_pedestrain_detection\label.txt', 'w')
    fw.writelines(newlines)

if __name__ == '__main__':
    crowd_convert()

