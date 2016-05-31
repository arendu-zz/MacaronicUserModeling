#!/bin/sh
set -e
source ./config.src
SIZE="50"
PRE="vocab."$SIZE
python train_ua_mp.py --users $TI_DIR/ti.$SIZE.users --ti $TI_DIR/ti.$SIZE --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_wiwj $FEATS_DIR/en-en.$SIZE.wi_wj.feats.mat  --phi_ed $FEATS_DIR/en-de.$SIZE.str.norm.mat  --phi_ped $FEATS_DIR/en-de.$SIZE.pron.norm.mat --cpu 4
