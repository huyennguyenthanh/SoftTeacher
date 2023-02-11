conda create --name mmdetection_v2 python=3.9
conda activate mmdetection_v2
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
python -m pip install openmim
mim install mmcv-full
make install
