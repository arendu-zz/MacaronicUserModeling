#!/bin/sh
set -e
TI_DIR="mturk-data"
FEATS_DIR="feats"
SIZE="50"
PRE="vocab."$SIZE
#python real_phi_test.py --ti $TI_DIR/ti.$SIZE --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_wiwj $FEATS_DIR/en-en.$SIZE.wi_wj.feats.mat  --phi_ed $FEATS_DIR/en-de.$SIZE.str.norm.mat  --phi_ped $FEATS_DIR/en-de.$SIZE.pron.norm.mat 
#PRE="vocab.50"
#python real_phi_test.py --ti $TI_DIR/ti.50 --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_wiwj $FEATS_DIR/en-en.50.wi_wj.feats.mat  --phi_ed $FEATS_DIR/en-de.50.str.norm.mat  --phi_ped $FEATS_DIR/en-de.50.pron.norm.mat 
python train_mp.py --ti $TI_DIR/ti.$SIZE --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_wiwj $FEATS_DIR/en-en.$SIZE.wi_wj.feats.mat  --phi_ed $FEATS_DIR/en-de.$SIZE.str.norm.mat  --phi_ped $FEATS_DIR/en-de.$SIZE.pron.norm.mat --cpu 4 --save_params $TI_DIR/outputs/ti.$SIZE.saved.params
