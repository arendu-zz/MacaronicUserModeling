#!/bin/sh
set -e
source ./config.src
SIZE="full"
PRE="vocab."$SIZE
python check_all_instances.py  --ti $TI_DIR/ti.$SIZE --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_wiwj $FEATS_DIR/en-en.$SIZE.wi_wj.feats.mat  --phi_ed $FEATS_DIR/en-de.$SIZE.str.norm.mat  --phi_ped $FEATS_DIR/en-de.$SIZE.pron.norm.mat 

