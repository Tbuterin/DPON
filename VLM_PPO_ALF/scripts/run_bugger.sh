# export ALFWORLD_DATA=~/alfworld-storage
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --config_file config_zero2.yaml --main_process_port 29336 \
    ../bugger_tester.py /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/jiaokechen/BACKUP/DPO4VLM/VLM_PPO_ALF/scripts/config_dpo.yaml \
    --env_name "AlfredThorEnv" \
    --alf_config ../alf-config.yaml \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 200 \
    --num-env-steps 12000 \
    --num_steps 128 \
    --grad-accum-steps 256 \
    --max-new-tokens 128 \
    --thought_prob_coef 0.2 \
    --use-gae True \
    --seed 123 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/jiaokechen/llava-mistral-sft-0926-1 \
    --use-lora True \
    --train-vision all \
    --reference_free True \
    # --wandb-project you_wandb_proj \
    # --wandb-run you_wandb_run \
    # --use-wandb \
    # --q4

    # thought_prob_coef details at /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/jiaokechen/DPO4VLM/VLM_PPO_ALF/a2c_ppo_acktr/llava_interface/interface.py
    # 1024
    # 103。2:3/128 5。 第8个31/128. 26ge 71/128. 40ge 31/128