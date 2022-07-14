import os

os.makedirs("result",exist_ok=True)
exp_names = ['Coffee','Scar','Scarf','Car','Easyship']
log_dir = [
    'log/Coffee/','log/Scar/','log/Scarf/','logs/Car/','logs/Easyship/'
]
ckpt_name = [
    'Coffee.th','Scar.th','Scarf.th','params.pkl','params.pkl'
]
testset_name = [
    "Coffee/imgs_test_all","Scar/imgs_test_all","Scarf/imgs_test_all",'test','test'
]

# Scar Coffee Scarf
os.chdir("./tensorf-myc")
for exp_id in [0,1,2]:
    exp = exp_names[exp_id]
    ckpt = log_dir[exp_id] + ckpt_name[exp_id]
    os.system(f"python train.py --config configs/{exp}.txt --render_only 1 --ckpt {ckpt}")
    test_path = log_dir[exp_id] + testset_name[exp_id]
    os.system(f"cp {test_path}/*.png ../result")
os.chdir("../")

# 对Easyship位参进行预处理，使用GARF修正后的位参
for split in ["train","val","test"]:    
    if not os.path.exists(f"./data_refine/Easyship/{split}"):
        os.system(f"cp -r ./data/Easyship/{split} ./data_refine/Easyship/{split}")

# Car Easyship
os.chdir("./jnerf-myc")
for exp_id in [3, 4]:
    exp = exp_names[exp_id]
    ckpt = log_dir[exp_id] + ckpt_name[exp_id]
    # os.system(f"python tools/run_net.py --config-file ./projects/ngp/configs/{exp}.py")
    test_path = log_dir[exp_id] + testset_name[exp_id]
    os.system(f"cp {test_path}/*.png ../result")
os.chdir("../")