# export ALFWORLD_DATA=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/jiaokechen/DPO4VLM/part_alf_data__
# Xvfb :0 -screen 0 1024x768x16 +extension GLX +render -noreset &
# # xhost +
# export DISPLAY=:0
# export MESA_GL_VERSION_OVERRIDE=3.3
# export MESA_GLSL_VERSION_OVERRIDE=330

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="0" accelerate launch --config_file config_zero2.yaml --main_process_port 29332 \
    ../main_alf.py /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/jiaokechen/DPON/VLM_PPO_ALF/scripts/config_dpo.yaml \
    --env_name "AlfredThorEnv" \
    --alf_config ../alf-config.yaml \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 200 \
    --num-env-steps 12000 \
    --num_steps $2 \
    --grad-accum-steps 256 \
    --max-new-tokens 1024 \
    --max_history_tokens 1024 \
    --thought_prob_coef $3 \
    --action_prob_coef $4 \
    --use-gae True \
    --seed 1001 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/jiaokechen/llava-mistral-sft-0926-1 \
    --use-lora True \
    --train-vision all \
    --max_pairs 1024 \
    --start_training_pair_nums 512 \
    --max_same_init_trajs 300 \
    --random_action_prob $6 \
    --check_grad False \
    --reference_free True \
    --cheat $5 \
    --task "$1" \
    --add_info "${10}" \
    --kl_ref $7 \
    --ref_kappa $8 \
    --use_action_prob_w $9 \
    --print_out_bug_story False \
    --saving_result_path ${11} \
    --task_wise ${12} \
    --test_sft ${13} \
    --use_return ${14} \
    # --wandb-project you_wandb_proj \
    # --wandb-run you_wandb_run \
    # --use-wandb \
    # --q4

    # thought_prob_coef details at /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/jiaokechen/DPO4VLM/VLM_PPO_ALF/a2c_ppo_acktr/llava_interface/interface.py
    # 1024
    # 103。2:3/128 5。 第8个31/128. 26ge 71/128. 40ge 31/128