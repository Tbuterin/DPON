import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
<<<<<<< HEAD
=======
from tqdm import tqdm
import pandas as pd
import copy
>>>>>>> 45fafb0... DPON

from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
import numpy as np  # jkc0924

# torch.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, max_new_tokens):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        #hard-code to cases of max_new_tokens being smaller than 32
        self.output_ids = torch.zeros(
            num_steps, num_processes, 2*max_new_tokens).long()
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.output_ids = self.output_ids.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, output_ids, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.output_ids[self.step].copy_(output_ids)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_pre2ds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            output_ids_batch = self.output_ids.view(-1,
                                              self.output_ids.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, output_ids_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ




<<<<<<< HEAD
# 240825tra # 20240830dic
import copy
import pickle
from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict
import pandas as pd
class TrajStorage:
    def __init__(self):
        self.tasks = {}  # 存储所有任务的字典，任务名是键，对应轨迹的字典是值

    def start_task(self, task_id):
        """开始一个新的任务"""
        if task_id in self.tasks:
            print(f"任务 {task_id} 已经存在。")
        else:
            self.tasks[task_id] = {}

    def start_trajectory(self, task_id, trajectory_id):
        """在指定任务下开始一条新的轨迹"""
        if task_id not in self.tasks:
            print(f"任务 {task_id} 不存在。")
        elif trajectory_id in self.tasks[task_id]:
            print(f"轨迹 {trajectory_id} 已经存在于任务 {task_id} 中。")
        else:
            self.tasks[task_id][trajectory_id] = []

    def add_point(self, task_id, trajectory_id, point):
        """向指定任务下的轨迹添加数据点"""
        if task_id not in self.tasks:
            print(f"任务 {task_id} 不存在。")
        elif trajectory_id not in self.tasks[task_id]:
            print(f"轨迹 {trajectory_id} 不存在于任务 {task_id} 中。")
        else:
            self.tasks[task_id][trajectory_id].append(copy.deepcopy(point))
=======

def compare_2_trajs(returna, returnb):
    """
    ta & tb: List(
        [obs, text_obs, text_action, success_rate]
    )
    """
    
    ##### TODO: 不同的判断方式🌟
    # reward_a, reward_b = max(float(ta[step_idx][3]), float(ta[-1][3])), max(float(tb[-1][3]), float(tb[step_idx][3]))
    # reward_a, reward_b = float(ta[step_idx][3]), float(tb[step_idx][3])
    # reward_a, reward_b = float(returna[step_idx]), float(returnb[step_idx])
    reward_a, reward_b = float(returna[0]), float(returnb[0])


    # if reward_a > reward_b and reward_a >= 0:
    #     return "better"
    # elif reward_a < reward_b and reward_b >= 0:
    #     return "worse"
    # else:
    #     print("success rate same")
    #     return "same"
    if reward_a > reward_b:
        return "better"
    elif reward_a < reward_b:
        return "worse"
    else:
        # print("success rate same")
        return "same"


def get_preference_data(preference, statA, statB, history_horizon=3):
    """
    这个函数用于将轨迹对转换成prompt + chosen/rejected action的方式返回
    Input:
        preference: str "better","worse"
        diff_idx: int 第一个不同元素的索引
        traA & traB: List([obs, text_obs, text_action, success_rate, reward, prompt], ...) 添加action_log_prob
    
    Output:
        pre_prompt
        pre_better
        pre_worse
        obs
    """
    # @TODO: 这里只考虑了text_obs相同，历史动作就一定相同的情况🌟
    # @TODO: 这里的prompt包含了历史轨迹。这里需要简化🌟
    # 真正动作不同的应该是第diff_idx - 1个动作
    # if diff_idx > 1:
    #     text_obs_action_pairs = [arr[1] + "\n" + arr[2] for arr in traA[:diff_idx - 1]]  # 修改为diff_idx - 1, 这里非常重要, 不是diff_idx - 2🌟
    # else:
    #     text_obs_action_pairs = []

    # text_obs_action_pairs = text_obs_action_pairs[-history_horizon:]
    text_obs_action_pairs_pre = []
    text_obs_action_pairs_rej = []
    if preference == "better":
        text_obs_action_pairs_pre.append(statA[5]) # 该步的prompt要添加
        text_obs_action_pairs_rej.append(statB[5])
    elif preference == "worse":  # jkc0926
        text_obs_action_pairs_pre.append(statB[5])
        text_obs_action_pairs_rej.append(statA[5])
    else:
        raise ValueError(f"No such preference: {preference}.")  # jkc0926

        
    pre_prompt_text = '\n'.join(text_obs_action_pairs_pre)
    rej_prompt_text = '\n'.join(text_obs_action_pairs_rej)
    # print(f"\033[35mpre_prompt_text: {pre_prompt_text}\033[0m")

    if preference == "better":
        pre_better_text = statA[2]
        pre_worse_text = statB[2]
        pre_prob = statA[6]  # jkc0920

        pre_obs = statA[0]
        rej_obs = statB[0]
    else:
        pre_better_text = statB[2]
        pre_worse_text = statA[2]
        pre_prob = statB[6]  # jkc0920

        pre_obs = statB[0]
        rej_obs = statA[0]

    
    return pre_prompt_text, rej_prompt_text, pre_better_text, pre_worse_text, pre_obs, rej_obs, pre_prob  # jkc0920
    

class TrajBuffer(object):
    def __init__(self, args, max_pairs, num_processes, max_history_tokens, max_new_tokens, obs_shape, history_horizon=3, max_same_init_trajs=200, gamma=0.99):
        """
        better_sample = better_obs_batch, better_output_ids_batch
        """
        self.args = args
        self.max_pairs = max_pairs
        self.max_history_tokens = max_history_tokens
        self.max_new_tokens = max_new_tokens
        self.buffer = []

        self.pre_prompt = torch.zeros(max_pairs, num_processes, 2*max_history_tokens).long()
        self.rej_prompt = torch.zeros(max_pairs, num_processes, 2*max_history_tokens).long()
        self.pre_better = torch.zeros(max_pairs, num_processes, 2*max_new_tokens).long()
        self.pre_worse = torch.zeros(max_pairs, num_processes, 2*max_new_tokens).long()
        self.pre_better_obs = torch.zeros(max_pairs, num_processes, *obs_shape)
        self.pre_worse_obs = torch.zeros(max_pairs, num_processes, *obs_shape)  # @TODO: 看看obs是否一致
        self.pre_action_log_prob = torch.zeros(max_pairs, num_processes, 1)  # jkc0920
        self.history_horizon = history_horizon

        self.valid_pairs = 0  # 这个变量用于存储既有数据的数量
        self.saving_index = 0  # 这个变量用于循环更新buffer的存储变量
        self.current_traj_index = 0

        self.max_same_init_trajs = max_same_init_trajs
        self.gamma=gamma
    
    def to(self, device):
        self.pre_prompt = self.pre_prompt.to(device)
        self.rej_prompt = self.rej_prompt.to(device)
        self.pre_better = self.pre_better.to(device)
        self.pre_better_obs =self.pre_better_obs.to(device)
        self.pre_worse = self.pre_worse.to(device)
        self.pre_worse_obs = self.pre_worse_obs.to(device)
        self.pre_action_log_prob = self.pre_action_log_prob.to(device)
        
    def save(self, path="./"):
        saving_dict = copy.deepcopy(self.buffer)
        max_length = max(len(value) for value in saving_dict.values())
        try:
            for key, value in saving_dict.items():
                if len(value) < max_length:
                    # 如果数组较短，用None（或其他值）填充
                    padding_length = max_length - len(value)
                    saving_dict[key] = list(value) + [None] * padding_length
                elif len(value) > max_length:
                    # 如果数组较长，进行截断
                    saving_dict[key] = list(value)[-max_length:]
            
            saving_dict_simple = {k: [[sub_list[1],sub_list[2],sub_list[3],sub_list[4],sub_list[5]] for sub_list in v] for k, v in saving_dict.items()}
            df = pd.DataFrame(saving_dict_simple)
            df.to_excel(path, index=False)
        except Exception as e:
            print(f"\033[41msaving failed: {e}\033[0m")
            # 打开输出文件（创建或覆盖模式）
            with open(f'{path[:-5]}.txt', 'w') as file:
                # 使用 print 函数，并通过 file 参数重定向输出到文件
                print(saving_dict, file=file)
    
    def refresh(self):
        self.buffer = []
        self.current_traj_index = 0

    def start_traj(self):
        """
        这个函数的作用是启动一个新的轨迹。
        在main_alf.py运行时如果接收到done=True, 或开始新的update循环, 则调用此函数。
        根据轨迹的init_observation_text, 如果存在则添加新轨迹, 不存在则创建相应的KEY。
        更新当前轨迹的KEY和Index (Index用于定位trajs的List(), 指的是在trajs中的第几个traj)。
        Input: init_observation_text (Str)
        Output: -
        """
        self.buffer.append([])
        self.current_traj_index = len(self.buffer) - 1

    def get_history_data(self):
        """
        这个函数的作用是用于prompt生成的时候引入历史轨迹信息
        Input: -
        Output: Str 由历史text_obs和text_action组成文本段落
        """
        try:
            if len(self.buffer[self.current_traj_index]) > 0:  # jkc0904
                text_obs_action_pairs = ["text_observation: " + arr[1] + "\naction: " + arr[2] for arr in self.buffer[self.current_traj_index][-self.history_horizon:]]
            else:
                return ""
            text_history = '\n'.join(text_obs_action_pairs)
        except Exception as e:
            print(f"\033[31m{e}, please start a trajectory first.\033[0m")
            exit(1)
        return text_history

    def add_new_state(self, obs, text_obs, text_action, success_rate, reward=None, prompt=None, action_log_prob=0., task=None):  # jkc0926🌟
        """
        这个函数的作用是加入一组数据到当前轨迹
        Input:
            obs: tensor([1, 300, 300, 3])
            text_obs: str
            text_action: str
            success_rate: float(1)
            prompt: str 这个是每一步所使用的prompt
        Output: -
        """
        self.buffer[self.current_traj_index].append([obs, text_obs, text_action, success_rate, reward, prompt, action_log_prob, task])
    
    
    def get_returns(self, traj):
        """
        计算每条轨迹中每个状态的回报 (return)
        """
        returns = []
        G = 0  # 累计回报初始化为0
        for t in reversed(traj):
            reward = t[4]  # 获取reward
            G = reward + self.gamma * G  # 更新累计回报
            returns.append(G)

        returns.reverse()  # 反转回报列表，使其与原始轨迹的顺序一致
        return returns

    def get_pairs_data(self, tokenizer, pad_num=0.):
        """
        构造样本对数据
        从self.buffer遍历读取相同init_state的轨迹, 构造样本对;
        存储到self.pre_prompt、self.pre_better、self.pre_better_obs和self.pre_worse、self.pre_worse_obs中。
        Input: tokenizer
        Output: _
        """
        # 遍历字典buffer中所有初始状态相同的轨迹对
        valid = False
        self.valid_pairs = 0  # jkc0909🌟
        self.saving_index = 0

        # random_traj_idx = np.random.permutation(len(self.buffer))  # jkc0924
        for first_traj_idx in tqdm(range(len(self.buffer) - 1, 0, -1)):
            for second_traj_idx in range(first_traj_idx - 1, -1, -1):
                # traA, traB = self.buffer[random_traj_idx[first_traj_idx]], self.buffer[random_traj_idx[second_traj_idx]]  # jkc0924
                traA, traB = self.buffer[first_traj_idx], self.buffer[second_traj_idx]

                # 同一个任务jkc0926🌟🌟🌟🌟🌟
                if len(traA) < 1 or len(traB) < 1: continue
                skip_current_pair=False
                if self.args.task_wise:
                    if traA[0][7] != traB[0][7]: 
                        skip_current_pair=True
                if skip_current_pair: continue

                min_length = min(len(traA), len(traB))

                # random_index = np.random.permutation(min_length)
                for sub_start in range(0, min_length):
                # for sub_start in random_index:
                    traA_, traB_ = traA[sub_start], traB[sub_start]
                    valid = True
                    if self.args.use_return:
                        preference = compare_2_trajs(self.get_returns(traA[sub_start:]), self.get_returns(traB[sub_start:]))
                    else:
                        preference = compare_2_trajs([traA_[4]], [traB_[4]])  # jkc0923
                    # preference = compare_2_trajs([traA_[3]], [traB_[3]])
                    if preference == "same": 
                        continue
                    # print("valid")
                    # print("valid")
                    # pre_prompt_text, rej_prompt_text, pre_better_text, pre_worse_text, pre_obs, rej_obs, pre_prob
                    pre_prompt_text, rej_prompt_text, pre_better_text, pre_worse_text, pre_obs, rej_obs, pre_action_log_prob = get_preference_data(preference, traA_, traB_, history_horizon=self.history_horizon)  # jkc0920
                    # print(f"\033[46mPreference: {pre_prompt_text} \n {pre_better_text} \n \033[44m {rej_prompt_text} \n {pre_worse_text} \033[0m")
                    
                    pre_prompt = tokenizer_image_token(pre_prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                    pre_prompt[pre_prompt == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace
                    rej_prompt = tokenizer_image_token(rej_prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                    rej_prompt[rej_prompt == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace
                    
                    pre_better = tokenizer(pre_better_text).input_ids
                    pre_worse = tokenizer(pre_worse_text).input_ids

                    # 广播到max_new_tokens的长度
                    if len(pre_better) < 2 * self.max_new_tokens:
                        pre_better += [pad_num] * (2 * self.max_new_tokens - len(pre_better))  # jkc0911
                    if len(pre_worse) < 2 * self.max_new_tokens:
                        pre_worse += [pad_num] * (2 * self.max_new_tokens - len(pre_worse))  # jkc0911
                    if pre_prompt.size()[-1] < 2 * self.max_history_tokens:
                        pre_prompt = torch.cat((pre_prompt, -100 * torch.ones(1, 2 * self.max_history_tokens - pre_prompt.size()[-1])), dim=1)  # jkc0921
                    if rej_prompt.size()[-1] < 2 * self.max_history_tokens:
                        rej_prompt = torch.cat((rej_prompt, -100 * torch.ones(1, 2 * self.max_history_tokens - rej_prompt.size()[-1])), dim=1)
                    
                    pre_better = pre_better[:2 * self.max_new_tokens]
                    pre_worse = pre_worse[:2 * self.max_new_tokens]
                    pre_prompt = pre_prompt[:, -2 * self.max_history_tokens:]  # jkc0904: 获取prompt中靠后的历史🌟  # @TODO: need to check
                    rej_prompt = rej_prompt[:, -2 * self.max_history_tokens:]
                    self.pre_prompt[self.saving_index % self.max_pairs].copy_(pre_prompt)
                    self.rej_prompt[self.saving_index % self.max_pairs].copy_(rej_prompt)
                    self.pre_better[self.saving_index % self.max_pairs].copy_(torch.tensor(pre_better))
                    self.pre_worse[self.saving_index % self.max_pairs].copy_(torch.tensor(pre_worse))
                    self.pre_better_obs[self.saving_index % self.max_pairs].copy_(pre_obs)
                    self.pre_worse_obs[self.saving_index % self.max_pairs].copy_(rej_obs)
                    self.pre_action_log_prob[self.saving_index % self.max_pairs].copy_(pre_action_log_prob)  # jkc0920
                    if self.valid_pairs < self.max_pairs - 1:
                        self.valid_pairs += 1  # 更新存储数据量
                    self.saving_index += 1  # 更新循环存储变量
                    
        if not valid:
            pass

        
    def feed_forward_generator(self, mini_batch_size=None):
        """
        由self.pre_prompt、self.pre_better和self.pre_worse读取并yield数据
        Output: prompt, better_sample和worse_sample的样本生成器--generator()
        """
        num_samples = self.valid_pairs
        print(f"\033[44mForward: mini_batch_size: {mini_batch_size}, valid_pairs: {self.valid_pairs}\033[0m")


        sampler = BatchSampler(
            SubsetRandomSampler(range(num_samples)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            pre_obs_batch = self.pre_better_obs[:-1].view(-1, *self.pre_better_obs.size()[2:])[indices]
            rej_obs_batch = self.pre_worse_obs[:-1].view(-1, *self.pre_worse_obs.size()[2:])[indices]
            pre_prompt_batch = self.pre_prompt[:-1].view(-1, self.pre_prompt.size()[-1])[indices]
            rej_prompt_batch = self.rej_prompt[:-1].view(-1, self.rej_prompt.size()[-1])[indices]
            pre_better_batch = self.pre_better[:-1].view(-1, self.pre_better.size()[-1])[indices]
            pre_worse_batch = self.pre_worse[:-1].view(-1, self.pre_worse.size()[-1])[indices]
            pre_action_log_prob = self.pre_action_log_prob[:-1].view(-1, self.pre_action_log_prob.size()[-1])[indices]   # jkc0920
            # print(f"\033[41m{type(pre_obs_batch)}\033[0m")
            
            yield pre_obs_batch, rej_obs_batch, pre_prompt_batch, rej_prompt_batch, pre_better_batch, pre_worse_batch, pre_action_log_prob  # jkc0920
>>>>>>> 45fafb0... DPON

    def get_trajectory(self, task_id, trajectory_id):
        """获取指定任务下的轨迹的全部数据"""
        if task_id in self.tasks and trajectory_id in self.tasks[task_id]:
            return self.tasks[task_id][trajectory_id]
        else:
            return []

    def get_all_tasks(self):
        """获取所有任务及其轨迹"""
        return self.tasks

    def delete_trajectory(self, task_id, trajectory_id):
        """删除指定任务下的轨迹"""
        if task_id in self.tasks and trajectory_id in self.tasks[task_id]:
            del self.tasks[task_id][trajectory_id]
        else:
            print(f"任务 {task_id} 或轨迹 {trajectory_id} 不存在。")

    def delete_task(self, task_id):
        """删除指定的任务及其所有轨迹"""
        if task_id in self.tasks:
            del self.tasks[task_id]
        else:
            print(f"任务 {task_id} 不存在。")
    
    def to(self, device):
        """将所有轨迹数据转移到指定设备"""
        for task_id, trajectories in self.tasks.items():
            for trajectory_id, points in trajectories.items():
                self.tasks[task_id][trajectory_id] = [point.to(device) for point in points]
    
    def save_to_file(self, file_path):
        """将所有任务及其轨迹保存到本地文件"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.tasks, f)
        print(f"\033[32m数据已保存到 {file_path}\033[0m")

    def load_from_file(self, file_path):
        """从本地文件加载任务及其轨迹"""
        with open(file_path, 'rb') as f:
            self.tasks = pickle.load(f)
        print(f"\033[32m数据已从 {file_path} 加载")

    def to_dataset_dict(self):
        """将所有任务及其轨迹转换为DatasetDict格式"""
        dataset_dict = {}
        for task_id, trajectories in self.tasks.items():
            data = []
            for trajectory_id, points in trajectories.items():
                for point in points:
                    data.append({
                        "task_id": task_id,
                        "trajectory_id": trajectory_id,
                        "point": point
                    })
            dataset = Dataset.from_pandas(pd.DataFrame(data))
            # 这里的键名可以根据需要调整，例如使用任务ID
            dataset_dict[task_id] = dataset
        return DatasetDict(dataset_dict)



if __name__ == "__main__":
    traj_storage = TrajStorage()

    # 开始新任务
    traj_storage.start_task("task1")

    # 在任务下开始新的轨迹
    traj_storage.start_trajectory("task1", "traj1")

    # 添加数据点
    traj_storage.add_point("task1", "traj1", {"step": 1, "obs": "you are in a bedroom"})
    traj_storage.add_point("task1", "traj1", {"step": 2, "obs": "you are in a livingroom"})

    # 获取指定任务下的轨迹的全部数据
    print("任务 task1 下的轨迹 traj1 的全部数据:")
    trajectory = traj_storage.get_trajectory("task1", "traj1")
    for point in trajectory:
        print(point)
    data = traj_storage.to_dataset_dict()
    print(f"\033[33m{data}\033[0m")
