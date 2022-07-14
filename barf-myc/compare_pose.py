import numpy as np
import json

from util_vis import *
import camera
import jittor as jt
import easydict

exp = "Easyship"
generate_method = 'trans'
val_path = f"../data/{exp}/"+"transforms_val.json"
new_val_path = f"../data_refine/{exp}/"+"transforms_val.json"

test_path = f"../data/{exp}/"+"transforms_test.json"
new_test_path = f"../data_refine/{exp}/"+"transforms_test.json"

with open(val_path,"r") as file:
    val = json.load(file)
with open(new_val_path,"r") as file:
    new_val = json.load(file)
all_trans = []
ref = []
pose = []
pose_new = []
frames = val["frames"]
new_frames = new_val["frames"]
for i,f in enumerate(frames):
    a = np.array(f["transform_matrix"])
    new_a = np.array(new_frames[i]["transform_matrix"])
    trans = new_a @ np.linalg.inv(a)
    all_trans.append(camera.pose.invert(jt.array(trans[:-1])).numpy().tolist())
    pose.append(camera.pose.invert(jt.array(a[:-1])).numpy())
    pose_new.append(camera.pose.invert(jt.array(new_a[:-1])).numpy())
    ref.append(np.eye(4)[:3])
fig = plt.figure(figsize=(10,10))
cam_path = "./"

# 输出所有变换矩阵
# with open("./trans.json","w") as file:
#     json.dump(all_trans,file,sort_keys=True, indent=4, separators=(',', ': '))

# 生成procrustes分析得出的sim3
all_trans = jt.array(all_trans)
pose = jt.array(pose)
pose_new = jt.array(pose_new)
center = jt.zeros([1,1,3])
center_pred = camera.cam2world(center,pose_new)[:,0] # [N,3]
center_GT = camera.cam2world(center,pose)[:,0] # [N,3]
sim3 = camera.procrustes_analysis(center_GT,center_pred)

# 读取原测试参数
with open(test_path,"r") as file:
    test = json.load(file)

frames = test["frames"]
# 使用procustes
if generate_method == 'sim3':
    for f in frames:
        a = jt.array(f["transform_matrix"][:3])
        a = camera.pose.invert(a).unsqueeze(0)
        center = jt.zeros((1,1,3))
        center = camera.cam2world(center,a)[:,0] # [N,3]
        center_aligned = (center-sim3.t0)/sim3.s0@sim3.R*sim3.s1+sim3.t1
        R_aligned = a[...,:3]@sim3.R
        t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
        pose = camera.pose(R=R_aligned,t=t_aligned)
        a = camera.pose.invert(pose)
        a = [col.numpy().tolist() for col in a.squeeze(0)]
        a.append([0,0,0,1])
        f["transform_matrix"] = a
    print(all_trans)
    print(sim3)
# 使用变换矩阵的均值
else:
    trans = jt.mean(all_trans,0).numpy()
    # trans = np.linalg.inv(trans)
    for f in frames:
        a = np.array(f["transform_matrix"])
        b = np.matmul(trans, a)
        f["transform_matrix"] = [ list(col) for col in b]

test["frames"] = frames

with open(new_test_path,"w") as file:
    json.dump(test,file,sort_keys=True, indent=4, separators=(',', ': '))

# 可视化转换矩阵的分布
# opt = edict()
# opt.visdom = edict()
# opt.visdom.cam_depth = 0.5
# plot_save_poses_blender(opt,fig,jt.array(all_trans),jt.Var(ref),path=cam_path)

