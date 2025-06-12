import json
import os
import shutil
from tqdm import tqdm
import glob


##get_visa_generate_data

with open('data/visa_meta.json', 'r') as file:
    data = json.load(file)


train_data=data["train"]

output_path = "data/visa_generate/"
input_path="data/visa/"

for class_name in tqdm(train_data):
    
    class_output = output_path+class_name
    
    if not os.path.exists(class_output):
        os.makedirs(class_output)
        
    
    for file in train_data[class_name]:
        
        if file['anomaly']!=0:
            print("error")
        
        image_path=file['img_path']
        
        shutil.copy(input_path+image_path, class_output)
        
        
##get_mvtec_generate_data

output_path = "data/mvtec_generate/"
input_path= "data/mvtec"

class_names = ["bottle","cable","capsule","carpet","grid","hazelnut","leather","metal_nut","pill","screw","tile","toothbrush","transistor","wood"]

for class_name in class_names:
    
    source_files = glob.glob(input_path+"/"+class_name+"/train/good/*.png")
    target_path = output_path+class_name
    
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    for file in source_files:
    
        shutil.copy(file,target_path)
    

        
        
        
        
    
    
    
