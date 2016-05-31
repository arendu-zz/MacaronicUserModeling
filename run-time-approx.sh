#!/bin/sh
set -e
TI_DIR="mturk-data"
FEATS_DIR="feats"
SIZE="200"
PRE="vocab."$SIZE
python time_tests.py --ti $TI_DIR/ti.$SIZE --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_pmi_w1 $FEATS_DIR/en-en.$SIZE.pmi.w1.feats.mat.scaled.01 --phi_pmi $FEATS_DIR/en-en.$SIZE.pmi.feats.mat.scaled.01  --phi_ed $FEATS_DIR/en-de.$SIZE.str.norm.mat.scaled.01  --phi_ped $FEATS_DIR/en-de.$SIZE.pron.norm.mat.scaled.01 --cpu 1 --save_params $TI_DIR/outputs/ti.$SIZE.saved.params --history --session_history --use_approx_learning
