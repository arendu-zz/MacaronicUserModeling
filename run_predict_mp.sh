#!/bin/sh
set -e
TI_DIR="mturk-data"
FEATS_DIR="feats"
SIZE="200"
PRE="vocab."$SIZE

#python train_mp.py --ti $TI_DIR/ti.$SIZE --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_wiwj $FEATS_DIR/en-en.$SIZE.pmi.feats.mat.scaled  --phi_ed $FEATS_DIR/en-de.$SIZE.str.norm.mat.scaled  --phi_ped $FEATS_DIR/en-de.$SIZE.pron.norm.mat.scaled --cpu 1 --load_params $TI_DIR/outputs/ti.$SIZE.saved.params --save_predictions $TI_DIR/outputs/ti.$SIZE.predictions

python train_mp.py --ti $TI_DIR/ti.$SIZE --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de  --phi_pmi_w1 $FEATS_DIR/en-en.$SIZE.pmi.w1.feats.mat.scaled.01 --phi_pmi $FEATS_DIR/en-en.$SIZE.pmi.feats.mat.scaled.01  --phi_ed $FEATS_DIR/en-de.$SIZE.str.norm.mat.scaled.01  --phi_ped $FEATS_DIR/en-de.$SIZE.pron.norm.mat.scaled.01 --cpu 1 --load_params $TI_DIR/outputs/ti.$SIZE.saved.params --save_predictions $TI_DIR/outputs/ti.$SIZE.predictions.w1.fixed --history --session_history
echo save predictions to $TI_DIR/outputs/ti.$SIZE.predictions.w1.fixed