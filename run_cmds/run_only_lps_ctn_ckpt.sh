#source /root/anaconda3/bin/activate gyk
ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir $ROOT/snapshot
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        $ROOT/subnet/main.py --log log.txt --config $ROOT/config256.json \
        --mse_loss-factor 0 --lps_loss-factor 1.0 \
        --lmd-mode random --lmd-lower_bound 2 --lmd-upper_bound 16 \
        --test-interval 2000 \
#       --from_scratch