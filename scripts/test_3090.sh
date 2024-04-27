source /root/anaconda3/bin/activate gyk
ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT
python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config2048.json \
 --test-lmd 256 \
 --pretrain snapshot/FT_0.03_512_2/iter84000.model \
 --test-path data/kodak/inputs

# CUDA_VISIBLE_DEVICES=3 python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config2048.json --pretrain $ROOT/snapshot/iter565355.model
# CUDA_VISIBLE_DEVICES=3 python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config2048.json --pretrain $ROOT/snapshot/iter646120.model
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config512.json \
#     --pretrain $ROOT/snapshot/iter726885.model
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config1024.json \
#     --pretrain $ROOT/snapshot/iter726885.model
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config2048.json \
#     --pretrain $ROOT/snapshot/iter726885.model