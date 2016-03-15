#!/bin/sh
set -e
TI_DIR=$HOME"/Projects/macaronic-feature-extraction/training-instances"
FEATS_DIR=$HOME"/Projects/macaronic-feature-extraction/feats"
SIZE="200"
PRE="vocab."$SIZE
python train_mp.py --ti $TI_DIR/ti.$SIZE --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_wiwj $FEATS_DIR/en-en.$SIZE.wi_wj.feats.mat  --phi_ed $FEATS_DIR/en-de.$SIZE.str.norm.mat  --phi_ped $FEATS_DIR/en-de.$SIZE.pron.norm.mat --cpu 4 --load_params $TI_DIR/outputs/ti.$SIZE.saved.params --save_predictions $TI_DIR/outputs/ti.$SIZE.predictions
