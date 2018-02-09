cp -R ~/128 /tmp/
mkdir /tmp/tmp_model

module unload gcc
module load cudnn/7.0
module load cuda90/toolkit
source activate tf
python train.py
source deactivate

time_stamp=$(date +%m_%d_%H_%M)
mkdir "~/model-${time_stamp}"
cp /tmp/tmp_model "~/model-${time_stamp}"


