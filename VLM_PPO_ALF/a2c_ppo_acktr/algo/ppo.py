import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import accelerate
import copy


def remove_trailing_value(tensor, value=-100):
    """
    删除张量末尾所有指定值的元素
    
    :param tensor: 输入张量
    :param value: 要删除的值
    :return: 删除指定值后的张量
    """
        # 找到第一个不等于-100的元素位置
    non_100_indices = (tensor != value).nonzero(as_tuple=True)[1]

    # 如果有不等于-100的元素，则取最大索引加1（因为索引从0开始）
    if len(non_100_indices) > 0:
        max_idx = non_100_indices[-1] + 1
        tensor = tensor[:, :max_idx]
    
    return tensor


class PPO():
    def __init__(self,
                 actor_critic,
                 optimizer,
                 accelerator,
                 clip_param,
                 ppo_epoch,
                 mini_batch_size,
                 value_loss_coef,
                 entropy_coef,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param

        self.ppo_epoch = ppo_epoch

        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optimizer
        self.accelerator = accelerator

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_step = 0
        self.actor_critic.train()
        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                    advantages, self.mini_batch_size)
            for sample in data_generator:
                with self.accelerator.accumulate(self.actor_critic):
                    grad_step += 1
                    obs_batch, output_ids_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ = sample
                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs = self.actor_critic.evaluate_actions(
                        obs_batch, output_ids_batch)
                    #values and action_log_probs on two different devices!! because they come from two llava
                    if torch.isnan(action_log_probs).any():
                        continue
                    old_action_log_probs_batch = old_action_log_probs_batch.to(action_log_probs.device).view(-1)
                    adv_targ = adv_targ.to(action_log_probs.device)
                    value_preds_batch = value_preds_batch.to(values.device)
                    return_batch = return_batch.to(values.device)


                    ratio = torch.exp(action_log_probs -
                                    old_action_log_probs_batch)

                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ
                    ## adding a ratio clip, inspired by https://github.com/huggingface/trl/blob/5a233546ee48532eaeb24b89b8d0042147574688/trl/trainer/ppo_trainer.py#L1199
                    if torch.any(ratio > 10):
                        action_loss = -surr2.mean()
                    else:
                        action_loss = -torch.min(surr1, surr2).mean()
                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + \
                            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                        value_loss = 0.5 * torch.max(value_losses,
                                                    value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2).mean()

                    try:
                        assert not torch.isnan(value_loss), "value_loss is nan"
                        assert not torch.isnan(action_loss), "action_loss is nan"
                    except:
                        print("value/action loss is nan")
                        exit(1)
                    loss = value_loss * self.value_loss_coef+action_loss
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.actor_critic.parameters(),
                            self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()

        value_loss_epoch /= grad_step
        action_loss_epoch /= grad_step
        dist_entropy_epoch /= grad_step

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch



# jkc240830: DPO
class DPO():
    def __init__(self,
                 training_args,
                 policy_model,  # 修改: 原来的actor_critic改为policy_model，表示DPO中的策略模型
                 reference_model,  # 添加: 参考模型，用于计算DPO损失
                 optimizer,
                 accelerator,
                 beta,  # 修改: PPO的clip_param替换为DPO的beta参数，用于控制DPO损失中的温度系数
                 dpo_epoch,  # 修改: ppo_epoch替换为dpo_epoch，表示DPO的训练轮数
                 mini_batch_size,
                 max_grad_norm=None,
                 label_smoothing=0.0,  # 添加: 标签平滑参数，用于DPO损失计算
                 ipo=False,
                 reference_free=False, # 添加: 是否使用参考模型的标志
                 ref_regular=False,  # jkc0920
                 gamma=0.1,  # jkc0920
                 tknz = None,  # jkc0921
                 ):  

        self.policy_model = policy_model  # 将actor_critic改为policy_model: DPOPolicy
        self.reference_model = reference_model  # 保存参考模型（添加reference_model属性）
        self.mini_batch_size = mini_batch_size
        self.beta = beta  # 将clip_param替换为beta
        self.dpo_epoch = dpo_epoch  # 将ppo_epoch替换为dpo_epoch
        self.label_smoothing = label_smoothing  # 添加label_smoothing属性
        self.ipo = ipo
        self.reference_free = reference_free  # 添加reference_free属性

        self.optimizer = optimizer
        self.accelerator = accelerator
        self.max_grad_norm = max_grad_norm

        self.regular = ref_regular  # jkc0920
        self.gamma=gamma  # jkc0920

        self.tknz=tknz  # jkc0921

        self.training_args = training_args
        self.use_action_prob_w = training_args.use_action_prob_w

    def update(self, rollouts, check_grad=False):
        """
        进行DPO的参数更新。
        rollouts: 由RolloutStorage管理的收集的样本，包括成对的较好和较差的样本。
        """
        action_loss_epoch = 0  # DPO中不需要区分value和action loss，因此移除action_loss_epoch和dist_entropy_epoch
        # action_loss_epoch = 0
        grad_step = 0
        self.policy_model.train()  # 将actor_critic改为policy_model
        if self.training_args.print_out_bug_story:
            print(f">>>>>>这里开始update了，经计算有效样本对共{rollouts.valid_pairs}对")
        for e in range(self.dpo_epoch):
            data_step = 0
            data_generator = rollouts.feed_forward_generator(self.mini_batch_size)  # @TODO
            for pre_obs, rej_obs, pre_prompt, rej_prompt, better_sample, worse_sample, pre_action_log_prob in data_generator:  # jkc0920
                # jkc0921
                pre_prompt = copy.deepcopy(remove_trailing_value(pre_prompt))
                rej_prompt = copy.deepcopy(remove_trailing_value(rej_prompt))

                if self.training_args.print_out_bug_story:
                    print(f"检查一下win样本的prompt和lose样本的prompt的数据类型: {type(pre_prompt)}, {type(rej_prompt)} \
                          数据大小: {pre_prompt.shape}, {rej_prompt.shape}。")
                data_step += 1
                print(f"\033[35m{data_step} / {rollouts.valid_pairs}\033[0m")

                ######## 合法性 ########
                non_zero_better_elements = torch.nonzero(better_sample)
                non_zero_worse_elements = torch.nonzero(worse_sample)

                if self.training_args.print_out_bug_story:
                    print(f"检查下win动作和lose动作的数据类型: {type(better_sample)}, {type(worse_sample)} \
                          数据尺寸: {better_sample.shape}, {worse_sample.shape}, 尺寸的第二维度应该等于2 * max_new_tokens \
                          数据中非零元素的个数: {non_zero_better_elements.size(0)}, {non_zero_worse_elements.size(0)} \
                          ")
                # 计算非零元素的数量
                num_non_zero_elements = max(non_zero_better_elements.size(0), non_zero_worse_elements.size(0))
                if num_non_zero_elements < 6:
                    print(f"\033[43mGrad Safe Problem!!!\033[0m")
                    continue
                ######## 合法性 ########

                ######## loss backward debug ########
                if check_grad:
                    # print(f"\033[34m>>>OBS {obs.size()}, isnan: {torch.isnan(obs).any()}, detype: {obs.dtype}\033[0m")
                    print(f"\033[35m>>>PT {pre_prompt.size()}: {pre_prompt}, isnan: {torch.isnan(pre_prompt).any()}\033[0m")
                    print(f"\033[32m>>>good {better_sample.size()}: {better_sample}\033[0m")
                    print(f"\033[33m>>>bad {worse_sample.size()}: {worse_sample}\033[0m")
            
                with self.accelerator.accumulate(self.policy_model):  # 将actor_critic替换为policy_model
                    grad_step += 1
                    better_obs_batch, better_output_ids_batch = pre_obs, better_sample
                    worse_obs_batch, worse_output_ids_batch = rej_obs, worse_sample
                    if self.training_args.print_out_bug_story:
                        print(f"开始梯度反向传播。\
                              检查better_obs_batch(图像obs)以及worse的数据类型: {better_obs_batch.dtype},{worse_obs_batch.dtype}\n\
                              尺寸: {better_obs_batch.shape}, {worse_obs_batch.shape} \
                              检查better_output_ids_batch(和better_sample一致, 是win的thts+action采样)以及worse的数据类型:{type(better_output_ids_batch)}, {type(worse_output_ids_batch)} \
                              数据尺寸: {better_output_ids_batch.shape}, {worse_output_ids_batch.shape}")
                    # 评估策略模型在较好和较差样本上的对数概率
                    better_log_probs, better_action_log_probs = self.policy_model.evaluate_actions(better_obs_batch, better_output_ids_batch, INPUT_IDS=pre_prompt) # obs是图片，ids是文本
                    worse_log_probs, worse_action_log_probs = self.policy_model.evaluate_actions(worse_obs_batch, worse_output_ids_batch, INPUT_IDS=rej_prompt)  # 这两个是要加input_IDS的
                    
                    if self.training_args.print_out_bug_story:
                        print(f"经过了policy_model的evaluate_actions, 输出better_log_probs和worse即加权的thts&action概率, \
                              数据类型: {better_log_probs.dtype}, {worse_log_probs.dtype} \n\
                              数据内容: {better_log_probs}, {worse_log_probs}, \n\
                              数据梯度: {better_log_probs.requires_grad}, {worse_log_probs.requires_grad} \n\
                              顺带检查一下动作的单独的概率: \n\
                              数据类型: {better_action_log_probs.dtype}, {worse_action_log_probs.dtype} \n\
                              数据内容: {better_action_log_probs}, {worse_action_log_probs}")
                    # if self.tknz is not None:
                    #     try: print(f"\033[46mbetter action text: {self.tknz.batch_decode(better_output_ids_batch, skip_special_tokens=True)}\033[0m")
                    #     except Exception as e: print(f"\033[31mbetter action text: {e}\033[0m")
                    #     try: print(f"\033[35mprompt: {self.tknz.batch_decode(prompt, skip_special_tokens=True)}\033[0m")
                    #     except Exception as e: print(f"\033[31mprompt: {e}\033[0m")
                    # print(f"\033[33moutput_id: {better_output_ids_batch.shape}: {better_output_ids_batch}\033[0m")
                    # print(f"\033[43mINPUT_IDS: {prompt.shape}: {better_output_ids_batch}\033[0m")

                    if check_grad:
                        policy_model_requires_grad = any(param.requires_grad for param in self.policy_model.parameters())
                        print(f"\033[43mpolicy_model in update requires grad: {policy_model_requires_grad}\033[0m")
                        # 检查 log_probs 的梯度信息
                        print("\033[32mBetter log probs requires grad:", better_log_probs.requires_grad)
                        # print("\033[32mBetter log probs grad fn:", better_log_probs.grad_fn)
                        print("\033[32mWorse log probs requires grad:", worse_log_probs.requires_grad)
                        # print("\033[32mWorse log probs grad fn:", worse_log_probs.grad_fn)
                    
                    
                    # Forward pass for reference model (or use precomputed reference log probs)
                    with torch.no_grad():  # jkc0904
                        # print(f"\033[32mreference free: {self.reference_free}\033[0m")
                        if self.reference_free:
                            reference_chosen_logps = torch.zeros_like(better_log_probs)  # reference_free模式下，参考模型的log probs为0
                            reference_reject_logps = torch.zeros_like(worse_log_probs)
                        else:
                            reference_chosen_logps, better_action_log_probs = self.reference_model.evaluate_actions(better_obs_batch, better_output_ids_batch, INPUT_IDS=pre_prompt)
                            reference_reject_logps, worse_action_log_probs = self.reference_model.evaluate_actions(worse_obs_batch, worse_output_ids_batch, INPUT_IDS=rej_prompt)
                        
                        if self.training_args.print_out_bug_story:
                            print(f"这里没有梯度，检查ref_model的值是否正确。\n\
                                  reference_chosen_logps的数据类型:{reference_chosen_logps.dtype}, 尺寸: {reference_chosen_logps.shape}")

                    # print(f"\033[42mbetter_log_probs: {type(better_log_probs)}\nworse_log_probs: {type(worse_log_probs)}\n\
                    #       \033[44mpre_action_log_prob: {pre_action_log_prob}--{type(pre_action_log_prob)}\n\
                    #       \033[45mbetter_action_log_probs: {type(better_action_log_probs)}\n\
                    #       worse_action_log_probs: {type(worse_action_log_probs)}\033[0m")
                    # 计算DPO损失
                    if self.training_args.print_out_bug_story:
                        print(f">>>计算dpo_loss")
                    losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                        better_log_probs,  # grad
                        worse_log_probs,  # grad
                        reference_chosen_logps,
                        reference_reject_logps,
                        self.beta,
                        self.label_smoothing,
                        self.gamma, # jkc0920
                        pre_action_log_prob,  # jkc0920
                        better_action_log_probs,  # jkc0920
                        worse_action_log_probs,  # jkc0921
                        self.ipo,
                        self.reference_free,
                    )

                    if self.training_args.print_out_bug_story:
                        print(f"\033[34m至关重要的一步，检查loss:{losses}, 类型{type(losses)}, 维度{losses.shape}, 梯度{losses.requires_grad}\033[0m")

                    # 检查数据合法性
                    # print(f"\033[43mlosses is: {losses}, {torch.isnan(losses)}\033[0m")
                    try:
                        assert not torch.isnan(losses).any(), "loss contains nan"
                    except:
                        print("\033[31mloss contains nan\033[0m")
                        exit(1)

                    if check_grad:
                        print(f"\033[32mLosses requires grad: {losses.requires_grad}\033[0m")
                        print(f"\033[32mLosses grad_fn: {losses.grad_fn}\033[0m")

                    loss = losses.mean()

                    if self.training_args.print_out_bug_story:
                        print(f"losses取mean之后的loss: {loss}, 类型: {loss.dtype}, 维度: {loss.shape}, 梯度: {loss.requires_grad}")
                        print(f"检查结束<<<<<<")
                    if check_grad:
                        print(f"\033[32mLosses requires grad after mean: {losses.requires_grad}\033[0m")
                        print(f"\033[32mLosses grad_fn after mean: {losses.grad_fn}\033[0m")


                    # 反向传播和梯度更新
                    try:
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.policy_model.parameters(),
                                self.max_grad_norm
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        action_loss_epoch += loss.item()
                    except Exception as e:
                        for _ in range(2): print("\033[41m##############################!!!\033[0m")
                        print(f"\033[41m{e}: losses: {losses}, better_output_ids_batch: {better_output_ids_batch.shape}\n \
                              better_log_probs: {better_log_probs}\n \
                              better_action_log_probs: {better_action_log_probs}\033[0m")
                        for _ in range(2): print("\033[41m##############################!!!\033[0m")
        
        try:
            action_loss_epoch /= grad_step
            print(f"losses: {action_loss_epoch}, epoch: {e/self.dpo_epoch}")
        except Exception as e:
            print(f"\033[41m<<!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!>>\033[0m")
            print(f"\033[41m{e}\033[0m")
            print(f"\033[41m<<!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!>>\033[0m")

        return action_loss_epoch

    def dpo_loss(self, 
                 policy_chosen_logps, 
                 policy_rejected_logps,
                 reference_chosen_logps, 
                 reference_rejected_logps,
                 beta, 
                 label_smoothing=0.0, 
                 gamma=0.1, 
                 ref_action_log_prob=torch.tensor([[0.]]), 
                 better_action_log_prob=torch.tensor([[0.]]), 
                 worse_action_log_prob=torch.tensor([[0.]]),
                 ipo=False, 
                 reference_free=False):
        """
        计算DPO损失函数
        policy_chosen_logps: 策略模型对选择样本的对数概率
        policy_rejected_logps: 策略模型对未选择样本的对数概率
        reference_chosen_logps: 参考模型对选择样本的对数概率
        reference_rejected_logps: 参考模型对未选择样本的对数概率
        beta: DPO损失的温度参数
        label_smoothing: DPO损失的标签平滑参数
        ipo: 是否使用IPO损失
        reference_free: 是否忽略参考模型
        """


        if self.reference_free:
            ref_logratios = 0  # 如果不使用参考模型

        # jkc0922
        
        if self.use_action_prob_w:
            with torch.no_grad():
                action_chosen_ps = torch.exp(better_action_log_prob)   #🌟jkc0924 
                action_rejected_ps = torch.exp(worse_action_log_prob)  #🌟jkc0924
            # if torch.all(action_chosen_ps==0.) and torch.all(action_rejected_ps==0.):
            if torch.all(action_chosen_ps<0.3) and torch.all(action_rejected_ps<0.3):  # jkc0925
                pi_logratios = torch.tensor(0.2) * (- policy_chosen_logps - policy_rejected_logps)  # jkc0924
            else:
                pi_logratios = action_chosen_ps * policy_chosen_logps - action_rejected_ps * policy_rejected_logps

        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
       

        # pi_logratios = action_chosen_ps * policy_chosen_logps - action_rejected_ps * policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios
        # print(f"\033[32m logits: {logits} \033[0m")
        # print(f"\033[42m{F.logsigmoid(logits)}\033[0m")

        # 计算DPO损失
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
        # print(f"\033[31mlosses is: {losses}\033[0m")
        # print(f"这里检查一下losses在dpo_loss函数中计算出来后的合法性:{losses}, 类型{losses.dtype}, 尺寸: {losses.shape}, logits: {logits}")
        if self.regular:
            # a=ref_action_log_prob.to(better_action_prob.device)
            # print(f"\033[42maction_log_prob: {ref_action_log_prob.device}\033[0m")
            # print(f"\033[42mpolicy_chosen: {better_action_prob.device}\033[0m")
            
            # print(f"\033[42m{a.device}\033[0m")
            # reg_losses = torch.exp(better_action_prob) * (better_action_prob - ref_action_log_prob.detach())   # jkc0920
            # reg_losses = -better_action_prob * (torch.exp(ref_action_log_prob.detach()) / torch.exp(better_action_prob))
            reg_losses = gamma * torch.exp(ref_action_log_prob.detach()) * nn.MSELoss()(better_action_log_prob, ref_action_log_prob.detach()).to(losses.dtype)  # jkc0924
            # print("\033[36m##################################################")
            # print(f"policy_chosen_logps: {better_action_prob}")
            # print(f"\033[31mpolicy_reject_logos: {worse_action_prob}\033[36m")
            # print(f"ref_origin: {ref_action_log_prob}")
            # # print(f"{better_action_prob - ref_action_log_prob.detach()}")
            # print(f"regular loss: {reg_losses}{reg_losses.shape}")
            # print("##################################################\033[0m")
            reg_losses_cliped = torch.clamp(reg_losses, min=-1.0, max=1.0).to(losses.dtype)  # jkc0923
            # print(f"检查ref_losses, 即kl散度正则项的合法性: {reg_losses_cliped}, 类型: {reg_losses_cliped.dtype}, 尺寸: {reg_losses_cliped.shape}")
            print(f"\033[45mloss_before: {losses}, {losses.dtype}\033[0m")
            # losses += reg_losses_cliped.view(1) * gamma  # jkc0920
            print(f"reg_loss: {reg_losses_cliped.dtype}")
            losses += reg_losses_cliped.view(1)  # jkc0924
            # print(f"加上正则项后, 检查losses合法性: {losses}, 类型: {losses.dtype}, 尺寸: {losses.shape}")
            print(f"\033[45mloss_after: {losses}, {losses.dtype}\033[0m")
        # try:
        #     print(f"loss: {losses}, reg_loss_cliped: {reg_losses_cliped}, logits: {logits}, action_chosen_ps: {action_chosen_ps}, action_rejected_ps: {action_rejected_ps}, policy_chosen_logps: {policy_chosen_logps}, policy_rejected_logps: {policy_rejected_logps}")
        # except:
        #     try:
        #         print(f"loss: {losses}, logits: {logits}, action_chosen_ps: {action_chosen_ps}, action_rejected_ps: {action_rejected_ps}, policy_chosen_logps: {policy_chosen_logps}, policy_rejected_logps: {policy_rejected_logps}")
        #     except:
        #         try:
        #             print(f"loss: {losses}, logits: {logits}, policy_chosen_logps: {policy_chosen_logps}, policy_rejected_logps: {policy_rejected_logps}")
        #         except Exception as e:
        #             print(f"{e}")

        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

