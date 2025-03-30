MLS_WAV_DIR='/export/corpora7/CapSpeech-MLS'
LIBRITTSRMIX_WAV_DIR='/export/corpora7/CapSpeech_LibrittsRmix'
GIGASPEECH_WAV_DIR='/export/corpora7/CapSpeech-GigaSpeech'
COMMONVOICE_WAV_DIR='/export/corpora7/CapSpeech-CommonVoice'
OUTPUT_DIR="./output_train/"
TEMPORY_SAVE_TO_DISK="./audio_code_train/"
SAVE_TO_DISK="./dataset_train/"
WANDB_KEY=''
HUGGINGFACE_KEY=''

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

accelerate launch ./training/run_parler_tts_training.py \
    --model_name_or_path "parler-tts/parler-tts-mini-v1" \
    --feature_extractor_name "parler-tts/dac_44khZ_8kbps" \
    --description_tokenizer_name "google/flan-t5-large" \
    --prompt_tokenizer_name "google/flan-t5-large" \
    --report_to "wandb" \
    --wandb_key ${WANDB_KEY} \
    --token ${HUGGINGFACE_KEY} \
    --overwrite_output_dir true \
    --train_dataset_name "OpenSound/CapSpeech-PretrainSet" \
    --train_split_name "train" \
    --eval_dataset_name "OpenSound/CapSpeech-PretrainSet" \
    --eval_split_name "val" \
    --mls_dir ${MLS_WAV_DIR} \
    --librittsrmix_dir ${LIBRITTSRMIX_WAV_DIR} \
    --gigaspeech_dir ${GIGASPEECH_WAV_DIR} \
    --commonvoice_dir ${COMMONVOICE_WAV_DIR} \
    --max_eval_samples 96 \
    --per_device_eval_batch_size 32 \
    --target_audio_column_name "audio_path" \
    --description_column_name "caption" \
    --source_column_name "source" \
    --prompt_column_name "text" \
    --max_duration_in_seconds 20 \
    --min_duration_in_seconds 3 \
    --max_text_length 600 \
    --preprocessing_num_workers 32 \
    --do_train true \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 6 \
    --gradient_checkpointing false \
    --per_device_train_batch_size 4 \
    --learning_rate 0.0005 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.01 \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_steps 4000 \
    --logging_steps 200 \
    --freeze_text_encoder false \
    --per_device_eval_batch_size 4 \
    --audio_encoder_per_device_batch_size 24 \
    --dtype "float16" \
    --seed 456 \
    --output_dir ${OUTPUT_DIR} \
    --temporary_save_to_disk ${TEMPORY_SAVE_TO_DISK} \
    --save_to_disk ${SAVE_TO_DISK} \
    --dataloader_num_workers 32 \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 4000 \
    --save_steps 4000 \
    --group_by_length true
