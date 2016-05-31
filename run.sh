#!/bin/sh
set -e
source ./config.src
SIZE="50"
PRE="vocab."$SIZE
#python real_phi_test.py --ti $TI_DIR/ti.$SIZE --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_wiwj $FEATS_DIR/en-en.$SIZE.wi_wj.feats.mat  --phi_ed $FEATS_DIR/en-de.$SIZE.str.norm.mat  --phi_ped $FEATS_DIR/en-de.$SIZE.pron.norm.mat 
python real_phi_test.py --ti $TI_DIR/ti.$SIZE --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_wiwj $FEATS_DIR/en-en.$SIZE.butter.feats.mat  --phi_ed $FEATS_DIR/en-de.$SIZE.zeros.feat.mat  --phi_ped $FEATS_DIR/en-de.$SIZE.zeros.feat.mat

