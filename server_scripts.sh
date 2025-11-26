apt install python3-pip unzip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install imageio tqdm reedsolo scikit-learn

mkdir -p ~/datasets
mkdir -p ~/datasets/ready_to_use

curl -L -o ~/datasets/GBRASNET.zip https://www.kaggle.com/api/v1/datasets/download/zapak1010/bossbase-bows2
unzip ~/datasets/GBRASNET.zip -d ~/datasets
rm ~/datasets/GBRASNET.zip

git clone https://github.com/Mechetel/ZhuNet.git

python3 ~/ZhuNet/prepare_gbrasnet.py --source ~/datasets/GBRASNET --destination ~/datasets/ready_to_use/GBRASNET

cd ZhuNet
python3 Main.py
