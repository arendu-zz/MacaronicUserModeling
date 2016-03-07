#!/bin/sh
set -e
TI_DIR="mturk-data"
FEATS_DIR="feats"
PRE="vocab.200"
python real_phi_test.py --ti $TI_DIR/ti.200 --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_wiwj $FEATS_DIR/en-en.200.wi_wj.en-en2wi_given_wj.feats.mat  --phi_ed $FEATS_DIR/en-de.200.str.norm.mat  --phi_ped $FEATS_DIR/en-de.200.pron.norm.mat 
#PRE="vocab.50"
#python real_phi_test.py --ti $TI_DIR/ti.50 --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_wiwj $FEATS_DIR/en-en.50.wi_wj.feats.mat  --phi_ed $FEATS_DIR/en-de.50.str.norm.mat  --phi_ped $FEATS_DIR/en-de.50.pron.norm.mat 

