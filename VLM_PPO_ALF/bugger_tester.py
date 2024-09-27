from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()

import time
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import gym_cards
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.transforms import ToPILImage  # jkc
import matplotlib.pyplot as plt  # jkc

from a2c_ppo_acktr import algo, utils, rl_utils
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection, get_alfworld_prompt
from a2c_ppo_acktr.rl_utils import get_dpo_prompt  # jkc0904
# from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.storage import RolloutStorage, TrajBuffer  # jkc
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model

# For alfworld
from alf_utils import load_config_file, get_obs_image, ALF_ACTION_LIST, process_action, compute_reward, AlfEnv


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
## IMAGE_TOKEN_INDEX: 在处理图像或混合数据（例如文本和图像）时，用于指示图像数据在整个数据序列中的索引位置。
## DEFAULT_IMAGE_TOKEN: 表示一个默认的图像标记，用于在数据序列中插入图像的占位符。这通常是一个特殊的标记，用于与非图像数据（如文本）区分开来。
## DEFAULT_IM_START_TOKEN: 表示图像数据段的起始标记。这通常用于指示序列中图像数据的起始位置，方便模型在处理时正确地识别和处理图像数据。
## DEFAULT_IM_END_TOKEN: 表示图像数据段的结束标记。这与起始标记一起使用，用于定义图像数据在序列中的范围。

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import LlavaLlamaForCausalLM

from functools import partial
from typing import List, Optional
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM
import transformers

from tqdm import tqdm

import accelerate
from accelerate.state import AcceleratorState

import warnings
warnings.filterwarnings("ignore")

import os
import copy

# jkc edit for DPO
from a2c_ppo_acktr.model import DPOPolicy
from transformers import HfArgumentParser
from configs import (
    H4ArgumentParser,
    ModelArguments,
    DataArguments,
    RLArguments,
    DPOConfig,
    StepDPOConfig
)
from dataclasses import dataclass, field
import random
from a2c_ppo_acktr.llava_interface import dpo_llava_generate, dpo_llava_evaluate


def torch_init(args):
    print(f"\033[43mGLOBAL SEED: {args.seed}\033[0m")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        print(f"\033[32mCUDA Deterministic.\033[0m")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)  # 限制 PyTorch （在CPU上）只使用一个线程，通常用于避免多线程竞争导致的性能下降。

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    transformers.set_seed(args.seed)


def load_base_model(args):
    model_path = args.model_path
    cache_dir = args.cache_dir

    # 打印模型路径。如果路径中包含 lora，加载 LoRA 模型，并检查是否支持 8bit 或 4bit 量化。如果不包含 lora，则加载标准的 LLaVA 模型，可能使用 8bit 或 4bit 量化。
    print(f"Path of the model is {model_path}")
    if "lora" in model_path:
        base, tokenizer = load_lora_model(model_path, cache_dir=cache_dir)
        if args.q8 or args.q4:
            raise ValueError("Lora model does not support 8bit or 4bit quantization")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        if args.q8:
            print("8bit quantization")
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
        elif args.q4:
            q4_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                    )
            print("4bit quantization")
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
        else:
            base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    return base, tokenizer








def main():
    parser = H4ArgumentParser((RLArguments, ModelArguments, DataArguments, StepDPOConfig))   # jkc0829
    args, model_args, data_args, training_args = parser.parse()   # jkc0829

    ###############
    # torch settings
    ###############
    torch_init(args)

    #########################
    # load model and tokenizer
    #########################
    base, tokenizer = load_base_model(args)  # base: 创建的Llava模型
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps)  # 处理分布式训练和梯度累积  # TODO
    device = accelerator.device
    print(f"\033[33mUsing {device}.\033[0m")
    model_device = device
    base = base.to(model_device)  # jkc0904
    


    # # 0919
    # rmodel_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/model/llama-7b-hf"
    # rtokenizer = AutoTokenizer.from_pretrained(rmodel_path)
    # rmodel = AutoModelForCausalLM.from_pretrained(rmodel_path)
    # 将模型移动到 GPU（如果可用）
    # rmodel.to(model_device)

    # 模型推理
    # with torch.no_grad():
    #     output = rmodel.generate(**input_tokens, max_length=50)

    # 解码输出
    tt = torch.tensor([[1, 330, 10706, 1444, 264, 13903, 2930, 304, 396, 18278, 10895, 13892, 28723, 415, 13892, 5212, 10865, 28725, 10537, 28725, 304, 27057, 11194, 298, 272, 2930, 28742, 28713, 4224, 28723, 2223, 725, 28747, 28705, 0, 28705, 13, 11159, 460, 396, 7583, 297, 272, 10461, 28765, 12466, 18065, 350, 823, 14802, 28723, 3604, 3638, 349, 298, 17801, 272, 14405, 10487, 395, 272, 634, 4784, 1057, 28723, 995, 460, 835, 2078, 272, 2296, 2245, 5436, 302, 272, 1868, 6337, 28747, 5936, 28733, 28746, 19417, 298, 7379, 11978, 28725, 10461, 28765, 12466, 28808, 327, 3137, 28711, 28756, 28711, 1976, 460, 297, 272, 4986, 302, 264, 2003, 28723, 14828, 4377, 1401, 368, 28725, 368, 1032, 264, 2855, 28705, 28740, 28725, 264, 9431, 28705, 28740, 28725, 264, 22895, 28705, 28740, 28725, 264, 22895, 28705, 28750, 28725, 264, 22895, 28705, 28770, 28725, 264, 22895, 28705, 28781, 28725, 264, 22895, 28705, 28782, 28725, 264, 22895, 28705, 28784, 28725, 264, 27673, 28705, 28740, 28725, 264, 27673, 28705, 28750, 28725, 264, 27673, 28705, 28770, 28725, 264, 20170, 4230, 28705, 28740, 28725, 264, 27673, 28705, 28781, 28725, 264, 27673, 28705, 28782, 28725, 264, 27673, 28705, 28784, 28725, 264, 27673, 28705, 28787, 28725, 264, 27673, 28705, 28783, 28725, 264, 27673, 28705, 28774, 28725, 264, 281, 411, 457, 28705, 28740, 28725, 264, 27673, 28705, 28740, 28734, 28725, 264, 27673, 28705, 28740, 28740, 28725, 264, 27673, 28705, 28740, 28750, 28725, 304, 264, 27673, 28705, 28740, 28770, 5923, 28711, 28756, 28711, 11159, 3638, 349, 298, 28747, 913, 438, 14405, 10487, 916, 272, 634, 4784, 1057, 14303, 3604, 5055, 815, 1070, 6768, 302, 272, 1868, 4620, 460, 28747, 5936, 1644, 298, 2855, 28705, 28740, 28742, 13, 464, 1644, 298, 9431, 28705, 28740, 28742, 13, 464, 1644, 298, 22895, 28705, 28740, 28742, 13, 464, 1644, 298, 22895, 28705, 28750, 28742, 13, 464, 1644, 298, 22895, 28705, 28770, 28742, 13, 464, 1644, 298, 22895, 28705, 28781, 28742, 13, 464, 1644, 298, 22895, 28705, 28782, 28742, 13, 464, 1644, 298, 22895, 28705, 28784, 28742, 13, 464, 1644, 298, 27673, 28705, 28740, 28742, 13, 464, 1644, 298, 27673, 28705, 28750, 28742, 13, 464, 1644, 298, 27673, 28705, 28770, 28742, 13, 464, 1644, 298, 20170, 4230, 28705, 28740, 28742, 13, 464, 1644, 298, 27673, 28705, 28781, 28742, 13, 464, 1644, 298, 27673, 28705, 28782, 28742, 13, 464, 1644, 298, 27673, 28705, 28784, 28742, 13, 464, 1644, 298, 27673, 28705, 28787, 28742, 13, 464, 1644, 298, 27673, 28705, 28783, 28742, 13, 464, 1644, 298, 27673, 28705, 28774, 28742, 13, 464, 1644, 298, 281, 411, 457, 28705, 28740, 28742, 13, 464, 1644, 298, 27673, 28705, 28740, 28734, 28742, 13, 464, 1644, 298, 27673, 28705, 28740, 28740, 28742, 13, 464, 1644, 298, 27673, 28705, 28740, 28750, 28742, 13, 464, 1644, 298, 27673, 28705, 28740, 28770, 28742, 13, 464, 262, 16917, 28742, 13, 464, 5819, 14303, 3604, 2899, 1023, 347, 264, 3716, 7611, 1729, 297, 272, 2296, 5032, 28747, 28705, 13, 11873, 13, 28739, 1013, 4488, 1264, 25002, 7489, 28725, 6685, 574, 3638, 304, 1868, 3425, 28723, 8126, 28725, 3084, 264, 3707, 28733, 1403, 28733, 7691, 24685, 354, 272, 1679, 6768, 368, 927, 298, 1388, 8089, 548, 28705, 13, 28739, 1774, 1264, 25002, 276, 5055, 815, 1070, 2992, 7935, 13, 18800, 8602, 8048, 12738, 28747]]).to(model_device)
    t2 = torch.tensor([[    1,  8789,  3371,    13, 28751,    13, 28705,   345,  1013,  4488,
          1264,   345,  2198,   396,  7583,   297,   272, 10461, 28765, 12466,
         18065,   350,   823, 14802, 28725,   586,  3638,   349,   298, 17801,
           272, 14405, 10487,   916,   272,   634,  4784,  1057, 28723,   315,
           837,  5489,   297,   272,  4986,   302,   264,  2003,   395,  4118,
          6697,  1259,   390, 21137, 28725,   634,  2285, 28725, 28201, 28725,
          3924,   404, 28725,   304,   264, 20170,   541, 28723,  1791, 17700,
           586,  3638, 28725,   315,   927,   298,   576,   298,   272,  9431,
           970,   272,   634,  4784,  1057,   349,  5651,   304,   868,   913,
           438,   272, 14405, 10487,   916,   272,   634,  4784,  1057,  9191,
            13, 28705,   345,  1774,  1264,   345,  1644,   298,  9431, 28705,
         28740, 28739,    13, 28752,    13, 13940, 28832, 28705,     2,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0]])

    t3 = torch.tensor([[    1,     1, 28705,  8789,  3371,    13, 28751,    13, 28705,   345,
          1013,  4488,  1264,   345,  2198,   396,  7583,   297,   272, 10461,
         28765, 12466, 18065,   350,   823, 14802, 28725,   586,  3638,   349,
           298, 17801,   272, 14405, 10487,   916,   272,   634,  4784,  1057,
         28723,   315,   837,  5489,   297,   272,  4986,   302,   264,  2003,
           395,  4118,  6697,  1259,   390, 21137, 28725,   634,  2285, 28725,
         28201, 28725,  3924,   404, 28725,   304,   264, 20170,   541, 28723,
          1791, 17700,   586,  3638, 28725,   315,   927,   298,   576,   298,
           272,  9431,   970,   272,   634,  4784,  1057,   349,  5651,   304,
           868,   913,   438,   272, 14405, 10487,   916,   272,   634,  4784,
          1057,  9191,    13, 28705,   345,  1774,  1264,   345,  1644,   298,
          9431, 28705, 28740, 28739,    13, 28752,    13, 13940, 28832, 28705,
             2,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0]])
    
    # t_real = torch.
    t4 = torch.tensor([[28739, 1774, 1264]])
    t5 = torch.tensor([[345, 1774, 1264]])
    # print(type(input_tokens))
    # print(f"\033[33m{type(input_tokens.input_ids)}, {input_tokens.input_ids}\033[35m {torch.equal(input_tokens.input_ids,tempt)} \033[0m")
    # print(f"\033[32m{type(temp)}, {temp}")

    output_ref = tokenizer.batch_decode(t2, skip_special_tokens=True)
    print(f"\033[36m\n{output_ref}\033[0m")
    output_ref = tokenizer.batch_decode(t3, skip_special_tokens=True)
    print(f"\033[35m\n{output_ref}\033[0m")
    output_ref = tokenizer.batch_decode(t4, skip_special_tokens=True)
    print(f"\033[34m\n{output_ref}\033[0m")
    output_ref = tokenizer.batch_decode(t5, skip_special_tokens=True)
    print(f"\033[34m\n{output_ref}\033[0m")

    action='"action":'
    t = tokenizer(action).input_ids
    t_ = tokenizer("\n" + action).input_ids
    tt = tokenizer(action + " go to sink 1").input_ids
    ttt = tokenizer("\n"+action+"go to sink 1").input_ids
    print(t)
    print(t_)
    print(tt)
    print(ttt)
    exit(1)


    print(f"\033[32mModel created.\033[0m")

    base.config.max_length = 1024  # @TODO: 修改更大值，因为加入了历史数据
    print(f"\033[33mModel max context length:\033[0m{base.config.max_length}")
    base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter = args.pretrain_mm_adapter)
    image_processor = base.get_vision_tower().image_processor

    # 配置 LoRA，设置其超参数 r, lora_alpha, target_modules, lora_dropout 等。如果启用了 LoRA，则使用该配置更新基础模型。
    base_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=find_all_linear_names(base,args.train_vision),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.use_lora:
        base = get_peft_model(base, base_lora_config)


    ## Inputing Prompt here
    ###############
    ## 实例化环境
    ###############
    assert args.alf_config is not None, "Alfworld environment requires a config file"
    print(f"\033[33mCreating Env: {args.alf_config}\033[0m")
    print(f"\033[33mPath: {os.getenv('ALFWORLD_DATA')}\033[0m")
    envs = AlfEnv(args.alf_config)
    obs, infos = envs.reset(seed=args.seed)
    admissible_commands = list(infos['admissible_commands'])[0]

     #################### Traj Storage ####################
    # jkc0904
    trajs = TrajBuffer(training_args.max_pairs, args.num_processes, training_args.max_history_tokens, args.max_new_tokens, (300, 300, 3), history_horizon=training_args.history_horizon)

    # trajs.add_test_state(tokenizer)
    # print(f"\033[41m{type(infos['observation_text'])}: {infos['observation_text']}\033[0m")
    trajs.start_traj(infos['observation_text'][0])

    #################### Traj Storage End ####################


    # 生成提示词 @TODO:需要修改提示词, 加入历史数据🌟
    history = trajs.get_history_data()
    # qs = get_alfworld_prompt(envs, obs = infos['observation_text'], admissible_actions=admissible_commands, action_only = args.action_only_prompt)
    qs = get_dpo_prompt(envs, obs = infos['observation_text'], admissible_actions=admissible_commands, history=history, action_only = args.action_only_prompt)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()  # 使用对话模板构建对话并生成最终的提示文本。
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # print(f"\033[34m{prompt}\033[0m")

    # 使用 tokenizer_image_token 函数将提示文本转化为输入 ID，返回张量格式，并确保所有零值位置都被替换为特定的标记（259）。
    INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace

    if "alfred" in args.env_name.lower():
        projection_f = partial(lambda x: x)

    policy_model = DPOPolicy(tokenizer=tokenizer,
                             image_processor=image_processor,
                             base=base,
                             projection_f=projection_f,
                             INPUT_IDS=INPUT_IDS,
                             args=args)
    
    if not training_args.reference_free:
      ref_model = copy.deepcopy(policy_model) ##TODO
      for param in ref_model.parameters():
          param.requires_grad = False
    else:
      ref_model=None

    optimizer = optim.Adam(policy_model.base.parameters(), lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay)  # 余弦退火学习率调度器，随着训练过程逐渐减少学习率。
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1  # 设置 DeepSpeed 的训练微批大小为 1。

    policy_model, optimizer, lr_scheduler = accelerator.prepare(policy_model, optimizer, lr_scheduler) ##TODO

    #################################################################
    # 创建 DPO（Direct Preference Optimization）代理，用于强化学习的策略优化。
    #################################################################
    agent = algo.DPO(
            policy_model,
            ref_model,
            optimizer,
            accelerator,
            training_args.beta,
            args.ppo_epoch,
            args.mini_batch_size,
            args.max_grad_norm,
            training_args.label_smoothing,
            training_args.reference_free
            )


    for i in range(100):
      image_tensor = obs
      image_tensor = policy_model.process_obs(image_tensor)

      padded_output_ids, outputs, sum_log_probs, action_tokens_log_prob = dpo_llava_generate(policy_model.base, tokenizer, input_ids=INPUT_IDS, image_tensor=image_tensor, args=args)
      print(f"\033[3{str(i%5+2)}m{i}: {outputs}\033[0m")
      
      action_random = random.choice(admissible_commands)
      action = ['\"action\":' + action_random]
      look_flag = 0
      p = random.random()
      if p < 0.1:
        action = ['\"action\": look']
        look_flag = 1
        print(f"\033[35m{i}: {p}\033[0m")
      obs, reward, done, infos = envs.step(action)
      # if look_flag == 0:
      #   print(f"\033[33m{i}: \033[34m action {action}, obs, {info['observation_text'][0]}, reward, {reward}\033[0m")
      # else:
      #   print(f"\033[36m{i}: {p}\033[35m action {action}, obs, {info['observation_text'][0]}, reward, {reward}\033[0m")
      
      qs = get_dpo_prompt(envs, obs = infos['observation_text'], admissible_actions=admissible_commands, history=history, action_only = args.action_only_prompt)
      qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
      conv = conv_templates[args.conv_mode].copy()  # 使用对话模板构建对话并生成最终的提示文本。
      conv.append_message(conv.roles[0], qs)
      conv.append_message(conv.roles[1], None)
      prompt = conv.get_prompt()
      # print(f"\033[34m{prompt}\033[0m")

      # 使用 tokenizer_image_token 函数将提示文本转化为输入 ID，返回张量格式，并确保所有零值位置都被替换为特定的标记（259）。
      INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
      INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace

######################################################################################################################################################
    tokenizer_ori = AutoTokenizer.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/jiaokechen/Qwen2-7B-Instruct")

    INPUT_IDS = torch.tensor([[ 1057,    13,  1774, 28747, 28705,     1, 28705,   371,    13, 28739,
           362,  1322, 28713,  1264,   345,   657,   272,  3469, 28725,   315,
          1032,   264,  2003,   395,  4118,  6697,  1259,   390,   264,  2855,
         28725,   264,  9431, 28725, 28201, 28725,  3924,   404, 28725,   304,
           264, 20170,   541, 28723,  1791,  4160,   272,  3638,   302,  2526,
           438,   396, 14405,  9917,   916,   272,  9431, 21157, 28725,   315,
           927,   298,   576,   298,   272,  9431,   304,   868,   913,   354,
           272, 14405,  9917, 28723,  8469, 28725,   272,  5055,   815,  1070,
          2992,   349,   464,  1644,   298,  9431, 28705, 28740, 28742,   304,
           868,   464,  5819, 28742,   354,   272, 14405,  9917,  9191,    13,
         28739,  1774,  1264,   345,  1644,   298,  9431, 28705, 28740, 28739,
            13, 28752, 28705,     2, 28705,    13, 29494,    13, 11159,  5055,
           815,  1070,  6768,   302,   272,  1868,  4620,   460, 28747,  5936,
          1644,   298,  2855, 28705, 28740, 28742,    13,   464,  1644,   298,
         22895, 28705, 28740, 28742,    13,   464,  1644,   298, 22895, 28705,
         28750, 28742,    13,   464,  1644,   298, 22895, 28705, 28770, 28742,
            13,   464,  1644,   298, 22895, 28705, 28781, 28742,    13,   464,
          1644,   298, 22895, 28705, 28782, 28742,    13,   464,  1644,   298,
         22895, 28705, 28784, 28742,    13,   464,  1644,   298, 27673, 28705,
         28740, 28742,    13,   464,  1644,   298, 27673, 28705, 28750, 28742,
            13,   464,  1644,   298, 27673, 28705, 28770, 28742,    13,   464,
          1644,   298, 20170,  4230, 28705, 28740, 28742,    13,   464,  1644,
           298, 27673, 28705, 28781, 28742,    13,   464,  1644,   298, 27673,
         28705, 28782, 28742,    13,   464,  1644,   298, 27673, 28705, 28784,
         28742,    13,   464,  1644,   298, 27673, 28705, 28787, 28742,    13,
           464,  1644,   298, 27673, 28705, 28783, 28742,    13,   464,  1644,
           298, 27673, 28705, 28774, 28742,    13,   464,  1644,   298,   281,
           411,   457, 28705, 28740, 28742,    13,   464,  1644,   298, 27673,
         28705, 28740, 28734, 28742,    13,   464,  1644,   298, 27673, 28705,
         28740, 28740, 28742,    13,   464,  1644,   298, 27673, 28705, 28740,
         28750, 28742,    13,   464,  1644,   298, 27673, 28705, 28740, 28770,
         28742,    13,   464, 20985, 13494, 28705, 28740,   477,  9431, 28705,
         28740, 28742,    13,   464, 20985, 22594, 28705, 28740,   477,  9431,
         28705, 28740, 28742,    13,   464, 20985,   290,   786, 28705, 28740,
           477,  9431, 28705, 28740, 28742,    13,   464, 20985,   290,   786,
         28705, 28750,   477,  9431, 28705, 28740, 28742,    13,   464, 20985,
         14405, 10487, 28705, 28740,   477,  9431, 28705, 28740, 28742,    13,
           464, 20985,  6183,  5538, 28705, 28740,   477,  9431, 28705, 28740,
         28742,    13,   464, 20985,  4969, 28705, 28740,   477,  9431, 28705,
         28740, 28742,    13,   464, 20985,   284, 16935, 28705, 28740,   477,
          9431, 28705, 28740, 28742,    13,   464, 20985, 19958, 28705, 28740,
           477,  9431, 28705, 28740, 28742,    13,   464, 20985,   284, 16935,
         28705, 28750,   477,  9431, 28705, 28740, 28742,    13,   464,   262,
         16917, 28742,    13,   464,  5819, 28742,    13,   464,   720, 21928,
          9431, 28705, 28740, 14303,  3604,  2899,  1023,   347,   264,  3716,
          7611,  1729,   297,   272,  2296,  5032, 28747, 28705,    13, 11873,
            13, 28739,   362,  1322, 28713,  1264, 25002,  4478,  6685,   767,
           511,   368,  1032,   297,   272,  3469,  1413,   272,  2245,  5436,
         28725,   868,  9547,  1073,   684,   690,  2992,   298,  4160,   272,
          3638, 28723,   443,   548, 28705,    13, 28739,  1774,  1264, 25002,
           276,  5055,   815,  1070,  2992,  7935,    13, 18800,  8602,  8048,
         259, 259]])

    output_ids_ = torch.tensor([[  1,   1, 259,    2,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0]])
    
    output_ids = torch.tensor([[    1,     1, 28705,   371,    13, 28739,   362,  1322, 28713,  1264,
           345,   657,   272,  3469, 28725,   315,  1032,   264,  2003,   395,
          4118,  6697,  1259,   390,   264,  2855, 28725,   264,  9431, 28725,
         28201, 28725,  3924,   404, 28725,   304,   264, 20170,   541, 28723,
          1791, 17801,   272, 14405,  9917,   395,   272,  9431, 21157, 28725,
           315,   927,   298,   576,   298,   272,  9431,   304,  1527,   356,
           272, 21157, 28723,  8469, 28725,   272,  5055,   815,  1070,  2992,
           349,   464,  1644,   298,  9431, 28705, 28740, 28742,   304,   464,
           499,   356,   272,  9431, 21157,  4135,   548,    13, 28739,  1774,
          1264,   371,    13, 28739,  1644,   298,  9431, 28705, 28740,  1264,
         12682,    13, 28739,   499,   356,   272,  9431, 21157,  1264,  4729,
            13, 28752,    13, 28752, 28705,     2,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0]])

    # image_tensor = torch.rand([1, 300, 300, 3])
    # image_tensor = policy_model.process_obs(image_tensor)
    # # image_tensor = image_tensor.view(-1, image_tensor.size()[2:])
    # p = tokenizer.decode(INPUT_IDS.tolist()[0])
    # a = tokenizer.decode(output_ids_.tolist()[0])
    # b = tokenizer.decode(output_ids.tolist()[0])

    #     # 使用 torch.nonzero 提取非零元素的索引
    # non_zero_elements = torch.nonzero(output_ids_)

    # # 计算非零元素的数量
    # num_non_zero_elements = non_zero_elements.size(0)
    # # b = tokenizer_ori.decode(IDS[:5])
    # print(f"\033[33m{p}\033[0m")
    # print("\033[34m", a, num_non_zero_elements,"\033[0m")
    # print("\033[35m", b, "\033[0m")

    # action_log_prob, _ = dpo_llava_evaluate(policy_model=base,  # 这里传入的是 base
    #                                         input_ids=INPUT_IDS,
    #                                         output_ids=output_ids,
    #                                         image_tensor=image_tensor,
    #                                         temperature=args.temperature,
    #                                         thought_prob_coef=args.thought_prob_coef)

    # action_log_prob_, _ = dpo_llava_evaluate(policy_model=base,  # 这里传入的是 base
    #                                         input_ids=INPUT_IDS,
    #                                         output_ids=output_ids_,
    #                                         image_tensor=image_tensor,
    #                                         temperature=args.temperature,
    #                                         thought_prob_coef=args.thought_prob_coef)
    # print(f"\033[43m{action_log_prob}\033[0m")
    # print(f"\033[35m{action_log_prob.requires_grad}\033[0m")
    # print(f"\033[36m{action_log_prob.grad_fn}\033[0m")

    # print(f"\033[43m{action_log_prob_}\033[0m")
    # print(f"\033[35m{action_log_prob_.requires_grad}\033[0m")
    # print(f"\033[36m{action_log_prob_.grad_fn}\033[0m")


def env_main():
  from alf_utils import load_config_file, get_obs_image, ALF_ACTION_LIST, process_action, compute_reward, AlfEnv
  import random
  from tqdm import tqdm
  import copy
  envs = AlfEnv("./alf-config.yaml")
  
  # slis = [3701, 9898389, 74734,1298, 34278,8954769, 626667, 1001, 21839,895, 12139954, 34332, 656555, 9023492]
  slis = [1001]
  for seed in slis:
    obs, infos = envs.reset(seed=seed)
    random.seed(seed)
    print(f"\033[37mseed: {seed}\033[0m")
    for i in range(1000):
      admissible_commands = list(infos['admissible_commands'])[0]
      action = random.choice(admissible_commands)
      action_list = ["\"action\":" + action]
      # if i > 15:
      #     action_list = ["\"action\":" + infos_new['extra.expert_plan'][0][0]]
      #     print(f"\033[34m{action_list}\033[0m")
      env_last = envs
      obs, reward, done, infos_new = envs.step(action_list)
      envs = env_last

      # print(infos_new['extra.expert_plan'][0])
      # infos = infos_new
      # print(f"\033[42m{reward}, \033[32m{infos_new['goal_condition_success_rate']}\033[0m")
      print(f"\033[34m{infos_new}\033[0m")
      # if reward > 0:
      #   print(f"\033[34m{infos}\n\033[35m{action}\n\033[34m{infos_new}\n\033[32m{reward}\n\n\033[0m")
      # elif reward == -1:
      #   print(f"\033[31minvalid\033[0m")
      # else:
      #     # print(f"\033[32m{reward}\033[0m")
      #     pass

if __name__ == "__main__":
    main()
    # env_main()