#source /root/anaconda3/bin/activate gyk
ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir $ROOT/snapshot
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
        $ROOT/subnet/main.py --log log.txt --config $ROOT/config256.json --mse_loss-factor 1.0 --lps_loss-factor 0.05 \
        --lmd-mode random --lmd-lower_bound 8 --lmd-upper_bound 256 \
        --test-interval 1500 \
        --exp-name A800_0506_revised \
        --batch-per-gpu 2 \
        --test-dataset-path data/Kodak24/ \
        --from_scratch
        # --pretrain snapshot_A/A800_0505_revised/iter4.model
        # --testuvg
        # --pretrain snapshot_A/A800_0505_revised/iter4.model
#        --pretrain snapshot/archive/bpp_mse_iter164000.model
