ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT
CUDA_VISIBLE_DEVICES=3 python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config2048.json --pretrain $ROOT/snapshot/iter807650.model
# CUDA_VISIBLE_DEVICES=3 python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config2048.json --pretrain $ROOT/snapshot/iter565355.model
# CUDA_VISIBLE_DEVICES=3 python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config2048.json --pretrain $ROOT/snapshot/iter646120.model
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config512.json \
#     --pretrain $ROOT/snapshot/iter726885.model
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config1024.json \
#     --pretrain $ROOT/snapshot/iter726885.model
# CUDA_VISIBLE_DEVICES=0  python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config2048.json \
#     --pretrain $ROOT/snapshot/iter726885.model