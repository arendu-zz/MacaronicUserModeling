#!/bin/sh
set -e
source /home/arenduc1/Projects/MacaronicUserModeling/config.src
SIZE="full"
PRE="vocab."$SIZE
TEST_FILE="ti.dev"
TRAIN_FILE="ti."$SIZE
TRAIN_TYPE="ti.train.1"
SAVE_PREDICTIONS=$TI_DIR/data-splits/newest-predictions
SAVE_FOLDER=$TI_DIR/data-splits/newest-outputs
mkdir -p $SAVE_PREDICTIONS
python train_mp.py --ti $TI_DIR/data-splits/$TEST_FILE --end $TI_DIR/$PRE.en.lower --ded $TI_DIR/$PRE.de --phi_pmi_w1 $FEATS_DIR/en-en.$SIZE.pmi.w1.feats.mat.scaled.01 --phi_pmi $FEATS_DIR/en-en.$SIZE.pmi.feats.mat.scaled.01  --phi_ed $FEATS_DIR/en-de.$SIZE.str.norm.mat.scaled.01  --phi_ped $FEATS_DIR/en-de.$SIZE.pron.norm.mat.scaled.01   --cpu 5 --load_params $SAVE_FOLDER/$TRAIN_TYPE.saved.params.iter1 --save_predictions $SAVE_PREDICTIONS/$TEST_FILE.predictions.from.$TRAIN_FILE.$TRAIN_TYPE --history --session_history  > run.predict.$SIZE.$TEST_FILE.from.$TRAIN_FILE.$TRAIN_TYPE.log
#--phi_len $FEATS_DIR/en-de.$SIZE.length.mat.scaled.01 
#--egt $TI_DIR/egt.full.en.lower  
#--ti_tgf $TI_DIR/data-splits/$TEST_FILE.tgf  
