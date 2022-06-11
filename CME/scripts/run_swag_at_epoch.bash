export swag_DIR=../data/nlu/swag
EPOCHS=$2
MODELNAME=$3
CKPT=$4
TYPE=$5
MODE=$6
model_type=$7
tokenizer_name=$8
visEPOCHS=$9
batch=${10:-16}
percent=${11:-100}

if [ "$MODELNAME" == "bert_small" ];
then
    MODEL=../snap/vlm/wiki103_bert_small/lastcheckpoint
elif [ "$MODELNAME" == "roberta_small" ];
then
    MODEL=../snap/vlm/roberta_vlm_wiki103
elif [ "$MODELNAME" == "bert_base" ];
then
    MODEL=../snap/vlm/vlm_12L_768H_wiki
elif [ "$MODELNAME" == "roberta_base" ];
then
    MODEL=../snap/vlm/vlm_roberta_12L_768H_wiki
else
    MODEL=$MODELNAME
fi

if [ "$MODE" == "pretrain" ];
then
    model_name_or_path=$MODEL/checkpoint-epoch00$CKPT
    config_name=$MODEL/checkpoint-epoch00$CKPT
elif [ "$MODE" == "loading" ];
then
    model_name_or_path=$MODEL # loading
    config_name=$MODEL
else
    model_name_or_path=$MODEL # loading
    config_name=$MODEL
    MODEL=../data/nlu/snap/vlm/$MODEL
fi

for TASK_NAME in swag
do
    for PERCENT in $percent
    do
        CUDA_VISIBLE_DEVICES=$1 python model/run_swag_ima.py \
            --training_percentage $PERCENT \
            --model_type $model_type \
            --tokenizer_name=$tokenizer_name \
            --model_name_or_path $model_name_or_path \
            --do_train \
            --do_eval \
            --train_file $swag_DIR/train.csv \
            --predict_file $swag_DIR/val.csv \
            --save_steps -1 \
            --max_seq_length 126 \
            --per_gpu_eval_batch_size=$batch   \
            --per_gpu_train_batch_size=$batch   \
            --learning_rate 2e-5 \
            --num_train_epochs $EPOCHS.0 \
            --config_name $config_name \
            --output_dir $MODEL/$TASK_NAME/$batch\_$TYPE\_percent$PERCENT  \
            --overwrite_output_dir \
            --num_langvis_train_epochs $visEPOCHS \
            --unifylangvis
        fi
    done
done
