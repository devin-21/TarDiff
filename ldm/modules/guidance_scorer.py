
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
from scipy.stats import truncnorm  # [Added code: 引入truncnorm用于采样]
import math

import torch.nn.functional as F

import numpy as np


class GradDotCalculatorMedformer:
    def __init__(self, model, train_loader, criterion, device='cuda', normalize='l2',gd_scale=1, mmd_scale=1,
                 pos=True, neg=True):
        """
        Initialize calculator with cached training gradients sum
        
        Args:
            model: trained PyTorch model
            train_loader: DataLoader containing training data
            criterion: loss function
            device: computation device
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.model.to(device)
        self.criterion.to(device)
        self.normalize = normalize
        # Cache the sum of training gradients
        self.train_loader=train_loader
        self.positive_guidance = pos
        self.negative_guidance = neg
        self.cached_train_grads = self._compute_train_grad_sum(train_loader)
        self.gd_scale=gd_scale
        self.mmd_scale=mmd_scale

        
    def compute_mmd_grad(self, x: torch.Tensor, kernel='linear') -> torch.Tensor:

        
        y = torch.stack([x[0] for x in self.train_loader]).to(self.device)
        x_flat = x.reshape(x.size(0), -1)  # (B1, D*T)
        y_flat = y.reshape(y.size(0), -1).float()  # (B2, D*T)
        if kernel == 'linear':
            K_xx = torch.mm(x_flat, x_flat.t())  # (B1, B1)
            K_yy = torch.mm(y_flat, y_flat.t())  # (B2, B2)
            K_xy = torch.mm(x_flat, y_flat.t())  # (B1, B2)
        elif kernel == 'rbf':
            gamma = 1.0 / x_flat.size(-1)  # 带宽参数
            pairwise_xx = torch.cdist(x_flat, x_flat, p=2)  # (B1, B1)
            K_xx = torch.exp(-gamma * pairwise_xx ** 2)
            pairwise_yy = torch.cdist(y_flat, y_flat, p=2)  # (B2, B2)
            K_yy = torch.exp(-gamma * pairwise_yy ** 2)
            pairwise_xy = torch.cdist(x_flat, y_flat, p=2)  # (B1, B2)
            K_xy = torch.exp(-gamma * pairwise_xy ** 2)
        else:
            raise ValueError("Unsupported kernel type")
    
        m = x_flat.size(0)
        n = y_flat.size(0)
        mmd = (K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()) 
        mmd_grad=torch.autograd.grad(mmd, x)


        return self._normalize_gradients(mmd_grad)
    
    def _normalize_gradients(self, grads):
        """
        Normalize gradients using specified method
        """
        if self.normalize == 'l2':
            # Compute total L2 norm across all parameters
            total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
            return [g / (total_norm + 1e-6) for g in grads]
        elif self.normalize == 'l1':
            # Compute total L1 norm across all parameters
            total_norm = sum(g.abs().sum() for g in grads)
            return [g / (total_norm + 1e-6) for g in grads]
        else:
            return grads
        
    def _compute_train_grad_sum(self, train_loader):
        """Compute and cache the sum of all training samples' gradients"""
        
        # Initialize gradient sum
        params = list(self.model.parameters())
        total_loss = torch.tensor(0.0, device=self.device,)
        # Accumulate loss for all training samples
        # data=torch.tensor(train_loader[0])
        
        # label=torch.tensor(train_loader[1])
        train_loader = DataLoader(train_loader, batch_size=1, shuffle=False)
        for train_inputs, train_labels in tqdm(train_loader):
            if (train_labels == 0 and not self.negative_guidance) or (train_labels == 1 and not self.positive_guidance):
                continue

            
            train_inputs = train_inputs.to(self.device).to(torch.float32)
            train_labels = train_labels.to(self.device).to(torch.long) 

            mask = torch.full((train_inputs.shape[0],train_inputs.shape[1]), True, dtype=bool,device=self.device)
            outputs = self.model(train_inputs,mask,None,None)
            total_loss += self.criterion(outputs, train_labels) 
        
        # Get gradient of total loss
        start_time = time.time()
        grad_sum = torch.autograd.grad(total_loss, params,allow_unused=True)
        filtered = [(p, g) for p, g in zip(params, grad_sum) if g is not None]
        if not filtered:
            raise ValueError("No parameter received gradient!")
        filtered_params, filtered_grads = zip(*filtered)
        print(f"Gradient computation time: {time.time() - start_time:.2f}s")
        torch.cuda.empty_cache()
        return self._normalize_gradients(filtered_grads)
        #return filtered_grads
    def compute_gradient(self, test_sample, test_label):
        """
        Compute gradient of grad-dot using cached training gradients
        
        Args:
            test_sample: single test input tensor
            test_label: ground truth label for test sample
        """
        self.model.eval()
        test_sample = test_sample.transpose(2,1)
        # Prepare test sample
        test_sample = test_sample.detach().requires_grad_(True)
        test_sample = test_sample.to(self.device)
        #test_label = torch.tensor([test_label]).to(self.device)
        test_label = test_label.to(self.device)
        # Get test gradient
        mask = torch.full((test_sample.shape[0],test_sample.shape[1]), True, dtype=bool,device=self.device)
        test_output = self.model(test_sample,mask,None,None)
        test_loss = self.criterion(test_output, test_label)
        test_grads = torch.autograd.grad(test_loss, self.model.parameters(), create_graph=True,allow_unused=True)
        filtered = [(p, g) for p, g in zip(self.model.parameters(), test_grads) if g is not None]
        if not filtered:
            raise ValueError("No parameter received gradient!")
        filtered_params, filtered_grads = zip(*filtered)
        test_grads = self._normalize_gradients(filtered_grads)
        # Compute single dot product with cached gradients sum
        total_dot = sum((test_g * cached_g).sum() 
                       for test_g, cached_g in zip(test_grads, self.cached_train_grads))
        # Get gradient w.r.t test_sample
        grad_wrt_sample = torch.autograd.grad(total_dot, test_sample, create_graph=False)[0]
        #mmd_grad=self.compute_mmd_grad(test_sample)[0].squeeze(0).transpose(2,1)
        gd_grad= grad_wrt_sample.squeeze(0).transpose(2,1)

        torch.cuda.empty_cache()
        batch_size = gd_grad.shape[0]
        # 计算比例因子，使得采样结果的期望值大约为 gd_scale
        #scale_factor = self.gd_scale / math.sqrt(2 / math.pi)
        dynamic_scale = self.gd_scale
        # 从标准正态分布采样，然后取绝对值，得到正半段的标准正态分布采样 
        #dynamic_scale = torch.abs(torch.randn(batch_size, device=gd_grad.device)) * scale_factor
        
        # reshape 成 [batch_size, 1, 1] 以便与 gd_grad 广播相乘
        #dynamic_scale = dynamic_scale.view(-1, 1, 1)
        #dynamic_scale = self.gd_scale
        return dynamic_scale*gd_grad

    def compute_noise_gd(self, test_sample, test_label, t):
        """
        Compute gradient of grad-dot using cached training gradients
        
        Args:
            test_sample: single test input tensor
            test_label: ground truth label for test sample
        """
        self.model.eval()
        test_sample = test_sample.transpose(2,1)
        # Prepare test sample
        test_sample = test_sample.detach().requires_grad_(True)
        test_sample = test_sample.to(self.device)
        #test_label = torch.tensor([test_label]).to(self.device)
        test_label = test_label.to(self.device)
        # Get test gradient
        mask = torch.full((test_sample.shape[0],test_sample.shape[1]), True, dtype=bool,device=self.device)
        ddim_timesteps = torch.tensor([
            0, 10, 20, 31, 43, 55, 67, 80, 93, 106,
            119, 132, 145, 157, 169, 179, 187, 193, 197, 199
        ], device=t.device)  # 确保和 t 在同一个 device

        # 将 t 映射为最近的 ddim timestep 的索引
        # t shape: [B], ddim_timesteps shape: [20]
        t_ = t.unsqueeze(1)  # [B, 1]
        diff = torch.abs(ddim_timesteps - t_)  # [B, 20]
        t_idx = torch.argmin(diff, dim=1)  # [B]，表示最近的 ddim index
        test_output = self.model(test_sample,mask,None,None,t_idx = t_idx)
        test_loss = self.criterion(test_output, test_label)
        test_grads = torch.autograd.grad(test_loss, self.model.parameters(), create_graph=True,allow_unused=True)
        filtered = [(p, g) for p, g in zip(self.model.parameters(), test_grads) if g is not None]
        if not filtered:
            raise ValueError("No parameter received gradient!")
        filtered_params, filtered_grads = zip(*filtered)
        test_grads = self._normalize_gradients(filtered_grads)
        
        # Compute single dot product with cached gradients sum
        test_grads_filtered = []
        cached_train_grads_filtered = []
        # import pdb; pdb.set_trace()
        # for i , j in zip(test_grads, self.cached_train_grads):
        #     if i.shape == j.shape:
        #         test_grads_filtered.append(i)
        #         cached_train_grads_filtered.append(j)
                

        #         print(i.shape)
        #         print(j.shape)
        total_dot = sum((test_g * cached_g).sum() 
                       for test_g, cached_g in zip(test_grads[:74], self.cached_train_grads[:74]))
        # Get gradient w.r.t test_sample
        grad_wrt_sample = torch.autograd.grad(total_dot, test_sample, create_graph=False)[0]
        #mmd_grad=self.compute_mmd_grad(test_sample)[0].squeeze(0).transpose(2,1)
        gd_grad= grad_wrt_sample.squeeze(0).transpose(2,1)

        torch.cuda.empty_cache()
        batch_size = gd_grad.shape[0]
        # 计算比例因子，使得采样结果的期望值大约为 gd_scale
        #scale_factor = self.gd_scale / math.sqrt(2 / math.pi)
        dynamic_scale = self.gd_scale
        # 从标准正态分布采样，然后取绝对值，得到正半段的标准正态分布采样 
        #dynamic_scale = torch.abs(torch.randn(batch_size, device=gd_grad.device)) * scale_factor
        
        # reshape 成 [batch_size, 1, 1] 以便与 gd_grad 广播相乘
        #dynamic_scale = dynamic_scale.view(-1, 1, 1)
        #dynamic_scale = self.gd_scale
        # pdb.set_trace()
        return gd_grad*1e7
    def compute_influence(self, test_sample, test_label, t):
        """
        Compute gradient of grad-dot using cached training gradients
        
        Args:
            test_sample: single test input tensor
            test_label: ground truth label for test sample
        """
        self.model.eval()

        test_sample = test_sample.transpose(2,1)
        # Prepare test sample
        test_sample = test_sample.detach().requires_grad_(True)
        test_sample = test_sample.to(self.device)
        #test_label = torch.tensor([test_label]).to(self.device)
        test_label = test_label.to(self.device)
        # Get test gradient
        mask = torch.full((test_sample.shape[0],test_sample.shape[1]), True, dtype=bool,device=self.device)
        test_output = self.model(test_sample,mask,None,None,t_idx = t)
        test_loss = self.criterion(test_output, test_label)
        test_grads = torch.autograd.grad(test_loss, self.model.parameters(), create_graph=True,allow_unused=True)
        filtered = [(p, g) for p, g in zip(self.model.parameters(), test_grads) if g is not None]
        if not filtered:
            raise ValueError("No parameter received gradient!")
        filtered_params, filtered_grads = zip(*filtered)
        test_grads = self._normalize_gradients(filtered_grads)
        # Compute single dot product with cached gradients sum
        total_dot = sum((test_g * cached_g).sum() 
                       for test_g, cached_g in zip(test_grads, self.cached_train_grads))
        torch.cuda.empty_cache()
        total_dot = total_dot.detach().cpu().numpy()
        return total_dot
    def compute_per_class_grad_norm_stats(self):
        """
        针对 train_loader 中每个样本，
        1. 计算损失关于模型参数的梯度，并对所有参数的梯度按 L2 范数计算该样本的梯度范数；
        2. 根据样本所属类别对梯度范数进行分组，最后统计输出每个类别的平均值和标准差（mean ± std）。
        
        返回:
            stats: dict，其中 key 为类别标签，value 为一个二元组 (mean, std)
        """
        self.model.eval()
        grad_norms = {}
        # 为了逐个样本计算梯度，这里将数据加载器包装成 batch_size=1
        data_loader = DataLoader(self.train_loader, batch_size=1, shuffle=False)
        
        for inputs, labels in tqdm(data_loader, desc="Computing per-class grad norms"):
            inputs = inputs.to(self.device).to(torch.float32)
            labels = labels.to(self.device)
            mask = torch.full((inputs.shape[0], inputs.shape[1]), True, dtype=torch.bool, device=self.device)
            outputs = self.model(inputs, mask, None, None)
            loss = self.criterion(outputs, labels)
            # 计算关于模型参数的梯度
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=False, allow_unused=True)
            # 过滤掉没有梯度的参数
            filtered_grads = [g for g in grads if g is not None]
            if not filtered_grads:
                continue
            # 计算梯度的 L2 范数：先对每个参数的梯度求平方和，再全部相加后开根号
            grad_norm_sq = sum(torch.sum(g ** 2) for g in filtered_grads)
            grad_norm = torch.sqrt(grad_norm_sq)
            # 假设 batch_size 为1，则 labels 只有一个数值
            label_val = labels.item() if labels.numel() == 1 else labels.tolist()[0]
            if label_val not in grad_norms:
                grad_norms[label_val] = []
            grad_norms[label_val].append(grad_norm.item())
        
        stats = {}
        # 对每个类别计算均值和标准差
        for label_val, norm_list in grad_norms.items():
            mean = np.mean(norm_list)
            std = np.std(norm_list)
            stats[label_val] = (mean, std)
            print("Class {}: {:.4f} ± {:.4f}".format(label_val, mean, std))
        return stats

    def compute_classifier_guidance(self, test_sample, test_label):
        """
        向量化版本：对 batch 中所有样本一次性计算 ∇ₓ log p(y|x)

        输入:
            test_sample: shape [B, D, T]，注意是未 transpose 的形式
            test_label: shape [B] or [B, 1]
        
        输出:
            guidance_grad: shape [B, D, T]，每个样本的 ∇ₓ log p(y|x)
        """
        self.model.eval()
        
        # 保证 test_sample 为 [B, T, D]，因为 model 期望输入是 (B, T, D)
        test_sample = test_sample.transpose(2, 1).detach().requires_grad_(True).to(self.device)
        test_label = test_label.to(self.device)

        if test_label.ndim > 1:
            test_label = test_label.squeeze(-1)

        B = test_sample.shape[0]

        # 前向计算 logits
        mask = torch.full((B, test_sample.shape[1]), True, dtype=torch.bool, device=self.device)
        logits = self.model(test_sample, mask, None, None)  # shape: [B, C]

        # log-softmax 全部类别概率
        log_probs = F.log_softmax(logits, dim=1)  # shape: [B, C]

        # 提取每个样本的对应类别的 log p(y|x)
        log_selected = log_probs.gather(1, test_label.view(-1, 1)).squeeze(1)  # shape: [B]

        # 对所有 log_selected 一起求和，得到标量 loss
        loss = log_selected.sum()
        # 对 loss 关于输入 x 求梯度，就能一次性得到 ∇ₓ log p(y_i|x_i)（注意 sum 不影响每个样本的梯度）
        grad = torch.autograd.grad(loss, test_sample, create_graph=False, retain_graph=False, allow_unused=True)[0] 
       
        return  grad.transpose(2, 1) *  self.gd_scale


class GradDotCalculator:
    def __init__(self, model, train_loader, criterion, gd_scale=1,
                 device='cuda', normalize='l2'):
        """
        用于计算与缓存训练集梯度之和，并在需要时对测试样本做 grad-dot。
        
        Args:
            model: 已训练好的 PyTorch 模型 (RNN 或其他)
            train_loader: DataLoader，包含训练数据集
            criterion: 损失函数
            device: 计算设备 ('cuda' 或 'cpu')
            normalize: 对梯度进行归一化的方式，可选 'l2' / 'l1' / None
        """
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.device = device
        self.normalize = normalize
        self.gd_scale = gd_scale
        self.cached_train_grads = self._compute_train_grad_sum(train_loader)

    def _normalize_gradients(self, grads):
        """
        按指定方式对梯度列表进行归一化。
        grads: list of torch.Tensor (每个param对应一个梯度)
        """
        if self.normalize == 'l2':
            # L2 范数
            total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
            return [g / (total_norm + 1e-6) for g in grads]
        elif self.normalize == 'l1':
            # L1 范数
            total_norm = sum(g.abs().sum() for g in grads)
            return [g / (total_norm + 1e-6) for g in grads]
        else:
            # 不进行归一化
            return grads

    def _compute_train_grad_sum(self, train_loader):
        """
        计算并缓存“整个训练集”上损失函数的梯度之和（对 model.parameters()）。
        
        这里的做法是：把训练集的 loss 全都累加到一个 total_loss，再一次性做 autograd。
        在部分应用中，这相当于把所有样本都看成了一个 batch。
        如果数据量极大，需要小心内存占用；也可以分段累加梯度再加起来。
        """
        
        # 参数列表
        params = list(self.model.parameters())
        
        # 累加所有训练样本的 loss
        total_loss = torch.tensor(0.0, device=self.device)
        
        for inputs, labels in tqdm(train_loader, desc="Compute Train Grad Sum"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).to(torch.float32)
            # 允许对 inputs 求梯度
            
            outputs = self.model(inputs)  # 对 RNN 来说即 forward(inputs)
            loss = self.criterion(outputs, labels)
            total_loss += loss

        # 一次性对 total_loss 求梯度
        grad_sum = torch.autograd.grad(total_loss, params, allow_unused=True)
        filtered_grads = [(p, g) for p, g in zip(params, grad_sum) if g is not None]
        filtered_params, filtered_grads = zip(*filtered_grads)
        # 归一化（可选）
        grad_sum = self._normalize_gradients(filtered_grads)
        torch.cuda.empty_cache()
        return grad_sum

    def compute_gradient(self, test_sample, test_label):
        """
        对单个测试样本计算“grad-dot”：
          1. 先对测试样本的 loss 求梯度 (w.r.t. model.parameters)
          2. 与 cached_train_grads 做点积
          3. 再对这个点积对输入 test_sample 求梯度
          
        Args:
            test_sample: shape = [batch_size, seq_len, feature_dim]（视你的 RNN 而定）
            test_label:  shape = [batch_size, ...]，与模型输出匹配
        Returns:
            grad_wrt_sample: 对输入 test_sample 的梯度
        """
        test_sample = test_sample.transpose(2, 1)  # 转置为 [batch_size, seq_len, feature_dim]
        # 允许对 test_sample 求梯度
        test_sample = test_sample.to(self.device)
        test_sample.requires_grad_(True)
        
        test_label = test_label.to(self.device).to(torch.float32)

        # 前向 & 计算测试损失
        with torch.backends.cudnn.flags(enabled=False):
            test_output = self.model(test_sample)
        test_loss = self.criterion(test_output, test_label)
        
        # w.r.t. 模型参数的梯度
        test_grads = torch.autograd.grad(
            test_loss,                  # scalar loss
            self.model.parameters(),    # 求梯度的对象
            create_graph=True           # 需要继续求高阶梯度
        )

        filtered_grads = [(p, g) for p, g in zip(self.model.parameters(), test_grads) if g is not None]
        filtered_params, filtered_grads_test = zip(*filtered_grads)
        test_grads = self._normalize_gradients(filtered_grads_test)

        total_dot = sum(
            (tg * cg).sum() 
            for tg, cg in zip(test_grads, self.cached_train_grads)
        )
        
        grad_wrt_sample = torch.autograd.grad(total_dot, test_sample)[0]
        
        return grad_wrt_sample.transpose(2, 1) * 1e6
