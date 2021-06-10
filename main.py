import numpy as np
import glob
import cv2
from ocr import ocr, get_file_size , read_base_json
import pickle
import json


img_paths = glob.glob('img/*g')
json_path = glob.glob('via*.json')


print(f"find {len(img_paths)} photos")

bases = read_base_json()
bases.pop('_via_image_id_list', None)

datas= {}

n=0

for path in img_paths:
    print(path)
    size = get_file_size(path)
    filename = path[4:]
    key = filename+str(size)
    datas[key]={'filename':filename,'size':size,'regions':[],'file_attributes':{}}
    img = cv2.imread(path)
    try:
        texts = ocr(img)
    except:
        texts = False
#     print(texts)
    if not texts:
        continue
    for text in texts:
        for box in text[0:-1]:
            b = box[3]
            x = b[0]
            y = b[1]
            w = b[2]-b[0]
            h = b[3]-b[1]
            rect = {'name':'rect','x':int(x),'y':int(y),'width':int(w),'height':int(h)}
            region = {'shape_attributes': rect, 'region_attributes':{'text': box[0]}}
            datas[key]['regions'].append(region)
    
bases['_via_img_metadata'] = datas
with open('save.json','w') as f:
    json.dump(bases,f)
    
print('Finish')
print('save save.json')