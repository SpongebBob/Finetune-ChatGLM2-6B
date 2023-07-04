
LR=6e-6
DATE=0704


MASTER_PORT=8888


deepspeed --num_gpus=8 --master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --do_eval \
    --train_file belleMath-train1k.json \
    --validation_file belleMath-dev1K.json \
    --prompt_column conversations \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm2-6b \
    --output_dir ./output/adgen-chatglm-6b-ft-$LR-$DATE \
    --overwrite_output_dir \
    --max_length 762 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 12 \
    --predict_with_generate \
    --num_train_epochs 3 \
    --logging_steps 20 \
    --save_steps 1000 \
    --learning_rate $LR \
    --do_eval False \
    --fp16 True \
    --save_total_limit 5 \
