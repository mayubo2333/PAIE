if [ $# == 0 ] 
then
    SEED=42
    LR=2e-5
else
    SEED=$1
    LR=$2
fi

work_path=exps/ace05_concat/$SEED/$LR
mkdir -p $work_path

python engine.py \
    --model_type paie \
    --dataset_type ace_eeqa \
    --model_name_or_path facebook/bart-base \
    --role_path ./data/dset_meta/description_ace.csv \
    --prompt_path ./data/prompts/prompts_ace_concat.csv \
    --seed $SEED \
    --output_dir $work_path  \
    --learning_rate $LR \
    --batch_size 16 \
    --eval_steps 500  \
    --max_steps 10000 \
    --max_enc_seq_length 180 \
    --max_prompt_seq_length 50 \
    --bipartite