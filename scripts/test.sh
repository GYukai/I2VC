# Test fid calculation specifically for 4090
#source /root/anaconda3/bin/activate gyk
ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT

mkdir $ROOT/snapshot
accelerate launch --main_process_port 29501 --config_file config_single.yaml   \
        $ROOT/subnet/main.py --log log.txt --config $ROOT/config256.json --mse_loss-factor 0 --lps_loss-factor 1.0 \
        --lmd-mode random --lmd-lower_bound 2 --lmd-upper_bound 16 \
        --exp-name TEST \
        --batch-per-gpu 2 \
        --test-path data/Kodak24/kodak \
        --pretrain  snapshot/mark/lpips0.03.model \
        --testuvg \
        --test-lmd 256
#        --from_scratch