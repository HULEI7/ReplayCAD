import os 
import json
import glob
from PIL import Image
from tqdm import tqdm

input_path = "data/"

meta_json_path = input_path+"visa_meta.json"
outpath = "data/visa/replay_meta.json"

#生成新的meta.json

with open(meta_json_path, 'r') as file:
    meta_data = json.load(file)

test_data = meta_data["test"]

new_data= {"train":{},"test":test_data}
new_data["train"]["pipe_fryum"] = meta_data["train"]["pipe_fryum"]


subfolders = [f.name for f in os.scandir(input_path+"generate") if f.is_dir()]

for class_name in subfolders:
    
    class_train_data = []
    class_train_images = glob.glob(input_path+"generate/"+class_name+"/samples/*.jpg")
    
    for class_image_path in class_train_images:
        
        data_path="/".join(class_image_path.split("/")[-4:])
        
        single_data = {"img_path":data_path,"mask_path": "","cls_name": class_name,"specie_name": "","anomaly": 0}
        
        class_train_data.append(single_data)
        
    
    new_data["train"][class_name]=class_train_data
    
with open(outpath, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
    
    
    