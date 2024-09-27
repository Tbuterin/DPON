import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate
import torch.nn.init as init

<<<<<<< HEAD
=======
# jkc0830
from a2c_ppo_acktr.llava_interface import dpo_llava_generate, dpo_llava_evaluate
import re

>>>>>>> 45fafb0... DPON
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class VLMValue(nn.Module):
    """
    actually the base is also used for generation!
    """
    def __init__(self, base):
        super(VLMValue, self).__init__()
        self.base = base
        # hard-code to llama hidden size for the value head
        self.value_head = nn.Sequential(
            nn.Linear(4096, 1024), # First layer
            nn.ReLU(), # Non-linearity
            nn.Linear(1024, 512), # Second layer
            nn.ReLU(), # Non-linearity
            nn.Linear(512, 1) # Output layer
            ).to(base.device, dtype=torch.float16) # Move to specified device with dtype

    def forward(self,  input_ids, image_tensor):
        if image_tensor.size(0) != 1:
            input_ids = input_ids.broadcast_to(image_tensor.size(0), input_ids.size(-1))

        image_tensor = image_tensor.to(self.base.device, dtype = self.base.dtype)
        _, _, _, _, inputs_embeds, _ = self.base.prepare_inputs_labels_for_multimodal(input_ids.to(self.base.device), None, None, None, None, image_tensor)
        inputs_embeds = inputs_embeds.to(self.base.device, dtype = self.base.dtype)
        assert inputs_embeds.shape[1] > 256
        outputs = self.base(
            inputs_embeds = inputs_embeds,
            output_hidden_states=True)
        hidden_states = outputs.hidden_states
        values = self.value_head(hidden_states[-1][:, -1])
        return values


class VLMPolicy(nn.Module):
    def __init__(self, tokenizer,
                image_processor,
                value_model,
                args,
                INPUT_IDS,
                projection_f,
                base_kwargs=None):
        """
        projection_f: the postprocessing function to parse text action
        """
        super(VLMPolicy, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.value_model = value_model
        self.base = value_model.base
        self.INPUT_IDS = INPUT_IDS
        self.projection_f = projection_f

    def process_obs(self, obs):
        #process the observation with the image processor
        processed_images = obs
        return self.image_processor.preprocess(processed_images, return_tensors='pt')['pixel_values'].to(dtype=self.base.dtype)

    def act(self, inputs, deterministic=False, INPUT_IDS=None):
        image_tensor = self.process_obs(inputs)
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        value, output_ids, text_action, action_log_prob, action_tokens_log_prob = llava_generate(value_model = self.value_model,
                                                    tokenizer = self.tokenizer,
                                                    input_ids = INPUT_IDS,
                                                    image_tensor = image_tensor,
                                                    args = self.args)
        action = self.projection_f(text_action)
        return value, output_ids, action, action_log_prob, action_tokens_log_prob

    def get_value(self, inputs, INPUT_IDS=None):
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        image_tensor = self.process_obs(inputs)
        return self.value_model(input_ids = INPUT_IDS, image_tensor = image_tensor)

    def evaluate_actions(self, inputs, output_ids, INPUT_IDS=None):
        image_tensor = self.process_obs(inputs)
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        value, action_log_prob, _ = llava_evaluate(value_model = self.value_model,
                                        input_ids = INPUT_IDS,
                                        output_ids = output_ids,
                                        image_tensor = image_tensor,
                                        temperature = self.args.temperature,
                                        thought_prob_coef = self.args.thought_prob_coef)
        return value, action_log_prob
<<<<<<< HEAD
=======



################################
# jkc0830: policy model for DPO
################################
class DPOPolicy(nn.Module):
    def __init__(self, tokenizer,
                 image_processor,
                 base,  # 直接使用 base 作为 policy model
                 args,
                 INPUT_IDS,
                 projection_f,
                 base_kwargs=None,
                 check_grad=False):
        """
        projection_f: the postprocessing function to parse text action
        """
        super(DPOPolicy, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.base = base  # 初始化时传入 base
        self.INPUT_IDS = INPUT_IDS
        self.projection_f = projection_f
        self.check_grad = check_grad

    def process_obs(self, obs):
        # 使用 image_processor 处理观测数据
        processed_images = obs
        return self.image_processor.preprocess(processed_images, return_tensors='pt')['pixel_values'].to(dtype=self.base.dtype)

    def act(self, inputs, deterministic=False, INPUT_IDS=None):
        image_tensor = self.process_obs(inputs)
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        # 使用 dpo_llava_generate 生成策略，不再使用 value_model
        output_ids, text_action, action_log_prob, action_tokens_log_prob = dpo_llava_generate(policy_model=self.base,  # 这里传入的是 base
                                                                                          tokenizer=self.tokenizer,
                                                                                          input_ids=INPUT_IDS,
                                                                                          image_tensor=image_tensor,
                                                                                          args=self.args)
        action = self.projection_f(text_action)
        return output_ids, action, action_log_prob, action_tokens_log_prob  # 返回策略相关输出

    def evaluate_actions(self, inputs, output_ids, INPUT_IDS=None):
        base_requires_grad = any(param.requires_grad for param in self.base.parameters())
        if self.check_grad:
            print(f"\033[45m DPOPolicy.base requires grad: {base_requires_grad}\033[0m")
        image_tensor = self.process_obs(inputs)
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        # 使用 dpo_llava_evaluate 进行策略评估，不再使用 value_model
        sum_action_log_prob, act_prob = dpo_llava_evaluate(policy_model=self.base,  # 这里传入的是 base
                                            input_ids=INPUT_IDS,
                                            output_ids=output_ids,
                                            image_tensor=image_tensor,
                                            temperature=self.args.temperature,
                                            thought_prob_coef=self.args.thought_prob_coef,
                                            action_prob_coef=self.args.action_prob_coef)  # jkc0923
        if self.check_grad:
            print(f"\033[44maction_log_prob after llava_eval RG: {sum_action_log_prob.requires_grad}\033[0m")
        return sum_action_log_prob, act_prob  # 返回策略相关的 log 概率

    # # jkc0919
    # def evaluate_kl_actions(self, vlm_tokenizer, ref_tokenizer, ref_model, output_ids, img_tensor):

    #     thts_action_txt = vlm_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    #      # 定义正则表达式模式
    #     pattern = r'(?s)(.*)"action"\s*:\s*"([^"]+)"'

    #     # 使用 finditer 提取所有匹配项
    #     matches = re.search(pattern, str(thts_action_txt))

    #     if matches:
        
    #         thts_sentence = matches.group(1)  # 获取最后一个action之前的文本
    #         action_sentence = matches.group(2)  # 获取最后一个action之后的句子
    #         print(f"\033[41m{thts_sentence}\033[31m]\n{action_sentence}\033[0m")
    #         # remaining_text = thts_action_txt[last_match.end():]  # 获取最后一个action之后的剩余文本
    #         input_ids=vlm_tokenizer(thts_sentence, return_tensors="pt")
    #         output_ids=vlm_tokenizer('"action":' + action_sentence, return_tensors="pt")
            
    #         input_ids_ref=ref_tokenizer(thts_sentence, return_tensors="pt")
    #         output_ids_ref=ref_tokenizer('"action":' + action_sentence, return_tensors="pt")
    #         with torch.no_grad():
    #             ref_prob = dpo_kl_llava_evaluate(ref_model, input_ids_ref, output_ids_ref, img_tensor, self.args.temperature)
    #         prob = dpo_kl_llava_evaluate(self.base, input_ids, output_ids, img_tensor, self.args.temperature)

    #         ref_prob += 0.00001
    #         kl_div = torch.log(prob) * torch.log(prob/ref_prob)
    #         return kl_div
    #     else:
    #         print(thts_action_txt)
    #         raise Exception("\033[31m NO ACTION \033[0m")
 
>>>>>>> 45fafb0... DPON
