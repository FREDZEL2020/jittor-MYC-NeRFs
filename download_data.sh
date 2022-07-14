mkdir -p data
cd data
wget https://cloud.tsinghua.edu.cn/f/63016014a4ad410997f5/?dl=1 -O Jrender_Dataset.zip 
unzip Jrender_Dataset.zip
wget https://cloud.tsinghua.edu.cn/f/d998312699ca45068ab1/?dl=1 -O B_test.zip
unzip B_test.zip
mv ./B_test/Car/transforms_test.json ./Car/transforms_test.json
mv ./B_test/Easyship/transforms_test.json ./Easyship/transforms_test.json 
mv ./B_test/Scar/transforms_test.json ./Scar/transforms_test.json 
mv ./B_test/Coffee/transforms_test.json ./Coffee/transforms_test.json 
mv ./B_test/Scarf/transforms_test.json ./Scarf/transforms_test.json 
cd ..
cd ..