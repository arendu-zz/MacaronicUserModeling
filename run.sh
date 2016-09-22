#!/bin/sh
set -e
source ./config.src
SIZE="full"
PRE="vocab."$SIZE
TRAIN_FILE="ti."$SIZE
time python train.py  --ti $TI_DIR/$TRAIN_FILE  --tune $TI_DIR/$TRAIN_FILE --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_pmi_w1 $FEATS_DIR/en-en.$SIZE.pmi.w1.feats.mat.scaled.01 --phi_pmi $FEATS_DIR/en-en.$SIZE.pmi.feats.mat.scaled.01  --phi_ed $FEATS_DIR/en-de.$SIZE.str.norm.mat.scaled.01  --phi_ped $FEATS_DIR/en-de.$SIZE.pron.norm.mat.scaled.01    --history --session_history   --save_params small.run.saved.params --reg_param 0.01 --report_times
