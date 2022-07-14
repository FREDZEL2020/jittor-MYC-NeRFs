# 获取Easyship的新位参
cd ../barf-myc
python evaluate.py --model=garf --yaml=Easyship --resume --start=0 --data.sub_val=
cp ./output/GARF/Easyship/*.json ../data_refine/Easyship
python compare_pose.py
cd ..
