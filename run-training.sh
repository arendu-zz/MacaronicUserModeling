#!/bin/sh
set -e
. ~/.bashrc
source /home/arenduc1/Projects/MacaronicUserModeling/config.src
SIZE="full"
PRE="vocab."$SIZE
TRAIN_FILE="ti.train.1"
SAVE_FOLDER=$TI_DIR/data-splits/newest-outputs
mkdir -p $SAVE_FOLDER 
python train_mp.py --ti $TI_DIR/data-splits/$TRAIN_FILE --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_pmi_w1 $FEATS_DIR/en-en.$SIZE.pmi.w1.feats.mat.scaled.01 --phi_pmi $FEATS_DIR/en-en.$SIZE.pmi.feats.mat.scaled.01  --phi_ed $FEATS_DIR/en-de.$SIZE.str.norm.mat.scaled.01  --phi_ped $FEATS_DIR/en-de.$SIZE.pron.norm.mat.scaled.01  --cpu 5 --save_params $SAVE_FOLDER/$TRAIN_FILE.saved.params --history --session_history --reg_param 0.01 > run.5.$TRAIN_FILE.train.log

