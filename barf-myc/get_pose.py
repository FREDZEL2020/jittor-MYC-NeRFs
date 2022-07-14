import numpy as np
import json

val_path = "/root/dongyu/data/Easyship1/transforms_test1.json"
new_val_path = "/root/dongyu/data/Easyship1/transforms_test2.json"

trans = np.array([[ 0.9995968 ,  0.0017076  , 0.00525882 , 0.00974721],
 [-0.00146808,  0.99893606 , 0.00665028  ,0.04816975],
 [-0.00551117, -0.00671873,  0.9987132  , 0.04079207],
 [0,0,0,1]])

# trans = np.linalg.inv(trans)

with open(val_path,"r") as file:
    val = json.load(file)

frames = val["frames"]
for f in frames:
    a = np.array(f["transform_matrix"])
    b = np.matmul(trans, a)
    f["transform_matrix"] = [ list(col) for col in b]

val["frames"] = frames

with open(new_val_path,"w") as file:
    json.dump(val,file)