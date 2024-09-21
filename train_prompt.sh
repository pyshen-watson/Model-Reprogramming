##################### Dataset #####################

DATA_DIR="../data"
SRC_NAME="CF10" # CF10, IN10
SRC_SIZE=32 # 32 for SRC/CF10, 112 for SRC/IN10
TGT_SIZE=28 # 28 for CF10, 86 for IN10
DATASET_PARAMS="--data_dir $DATA_DIR --src_size $SRC_SIZE"

##################### Model #####################

MODEL_TYPE="DNN" # DNN, CNN, VGG or ResNet
WIDTH=2048 # 2048 for DNN, 512 for CNN, 32 for VGG and ResNet
MODEL_PARAMS="-m $MODEL_TYPE -w $WIDTH"

##################### Optimizer #####################

LEARNING_RATE=0.001
WEIGHT_DECAY=0.0
MAX_EPOCHS=150
PATIENCE=3
OPTIM_PARAMS="--learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY -M $MAX_EPOCHS -P $PATIENCE"

##################### Range #####################
LAYER_RANGE="2 4 5 6 7 9"
GROUP_RANGE="1"
DATASET_RANGE="stl10" # cifar10, svhn, stl10
LOSS_FN_RANGE="CE"

##################### Training #####################

for layer in $LAYER_RANGE; do
    for dataset in $DATASET_RANGE; do
        for loss_fn in $LOSS_FN_RANGE; do
            for group in $GROUP_RANGE; do

                gpu_id=0
                if [ "$loss_fn" = "CE" ]; then
                    gpu_id=1
                fi

                EXP_PARAMS="--dataset $dataset --loss_fn $loss_fn -l $layer -g $group"
                # WEIGHT_PARAMS="--weight_path weights/SRC/${MODEL_TYPE}/${SRC_NAME}-${loss_fn}/${MODEL_TYPE}-${layer}x${group}.pt"
                WEIGHT_PARAMS="--weight_path weights/SRC/${MODEL_TYPE}/${SRC_NAME}-${loss_fn}/${MODEL_TYPE}-${layer}.pt"
                params="$DATASET_PARAMS $MODEL_PARAMS $OPTIM_PARAMS $EXP_PARAMS $WEIGHT_PARAMS"

                python train_prompt.py --tgt_size $SRC_SIZE $params -G $gpu_id &
                # python train_prompt.py --tgt_size $TGT_SIZE $params -G $gpu_id -V &
                # python train_prompt.py --tgt_size $SRC_SIZE $params -G $gpu_id -F & # Cannot for IN10
                # python train_prompt.py --tgt_size $TGT_SIZE $params -G $gpu_id -V -F & # Cannot for IN10

            done
        done
    done    
wait 
done
