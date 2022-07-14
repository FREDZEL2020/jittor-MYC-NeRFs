# NeRF-myc MYC组 计图第二届人工智能挑战赛NeRF赛道二代码

## 项目简介
- 该项目一共使用了三种NeRF模型：
* 1. TensoRF
    * [Original](https://github.com/apchenstu/TensoRF) |  [Paper](https://arxiv.org/abs/2203.09517)
    * 实现了jittor框架下的TensoRF
    * 在TensoRF的基础上融合了NeRF++与Ref-NeRF
* 2. JNeRF
    * [Original](https://github.com/Jittor/JNeRF)
    * 基本上复用了原版JNeRF，进行了少量参数的调节
* 3. GARF
    * [Original](https://github.com/chenhsuanlin/bundle-adjusting-NeRF)
    * 实现了jittor框架下的BARF的代码结构（jittor框架下的BARF暂时无法复现原Pytorch下BARF的结果）
    * 在BARF的结构上实现了GARF，能够解决Inaccurate Camera Pose的问题，校准不精确的相机外参
    * 可以输出修正后的训练与验证集的相机外参，并一定程度上推测测试集的相机外参

## 环境配置
* 环境要求
```
jittor
python
numpy
```
* 配置环境脚本
```
bash install.sh
```

## 数据集
* 下载数据集（比赛B榜）
```
bash download_data.sh
```
* 数据集结构
```
# 原数据集
/data
    /Easyship
    /Car
    ...
# 修正位参后的数据集
/data_refine
    /Easyship
    ...
```

## 输出测试集
* 输出比赛B榜测试集到/result文件夹
```
python test.py
```

## 训练模型
### TensoRF
* 训练
```
python train.py --config ./configs/Scar.txt
```
* 输出测试图片
```
python train.py --config ./configs/Scar.txt --ckpt <path to ckpt> --render_only 1
```

### GARF
* 训练
```
python train.py --group=<GROUP> --model=garf --yaml=Easyship
```
* 输出训练集位参、校准验证集位参、并输出验证集图片与位参
```
python evaluate.py --group=<GROUP> --model=garf --yaml=Easyship --resume --start=0  --data.sub_val=
```
* 根据新旧验证集位参推测新测试集位参(需更换compare_pose.py文件中`transforms_xxx.json`文件路径)
```
python compare_pose.py
```

### JNeRF
* 请见/jnerf-myc/README.md