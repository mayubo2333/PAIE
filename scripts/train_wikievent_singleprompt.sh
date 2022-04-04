if [ $# == 0 ] 
then
    SEED=42
    LR=2e-5
else
    SEED=$1
    LR=$2
fi

work_path=exps/wikievent_singleprompt/$SEED/$LR
mkdir -p $work_path # make output dir

python -u engine.py \
    --model_type base \
    --dataset_type wikievent \
    --model_name_or_path ckpts/bart-base \
    --role_path ./data/dset_meta/description_wikievent.csv \
    --seed $SEED \
    --output_dir $work_path \
    --max_steps 10000 \
    --learning_rate $LR \
    --max_enc_seq_length 500 \
    --max_dec_seq_length 20 