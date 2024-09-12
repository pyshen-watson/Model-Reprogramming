##################### Dataset #####################

DATASET="imagenet10" # cifar10, imagenet10
DATA_DIR="../data/ImageNet10"
SRC_SIZE=112 # 32 for cifar10, 112 for imagenet10
DATASET_PARAMS="--dataset $DATASET --data_dir $DATA_DIR --src_size $SRC_SIZE"


##################### Model #####################

MODEL_TYPE="CNN" # DNN, CNN, VGG or ResNet
WIDTH=512 # 2048 for DNN, 512 for CNN, 32 for VGG and ResNet
MODEL_PARAMS="-m $MODEL_TYPE -w $WIDTH"


##################### Optimizer #####################

LOSS_FN="CE" # MSE, CE
LEARNING_RATE=0.001
WEIGHT_DECAY=0.001
OPTIM_PARAMS="--loss_fn $LOSS_FN --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY"



for seed in $(seq 42 44);do
    for group in $(seq 1 1);do
        for layer in $(seq 3 4);do
            params="$DATASET_PARAMS $MODEL_PARAMS $OPTIM_PARAMS -l $layer -g $group"
            gpu_id=$(((layer+group) % 2))
            python train_source.py $params -r $seed -G $gpu_id &
        done
        wait
    done
done

