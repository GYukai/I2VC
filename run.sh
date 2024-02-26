source /root/anaconda3/bin/activate gyk
ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir $ROOT/snapshot
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        $ROOT/subnet/main.py --log log.txt --config $ROOT/config256.json \
        # --from_scratch
        # --pretrain $ROOT/snapshot/iter484590.model 