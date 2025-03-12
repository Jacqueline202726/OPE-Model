import numpy as np
import torch
import torch.nn as nn

from Models import Expert, Tower

class CGC_OPE(nn.Module):
    """
    多任务学习的专家网络（Conditional Gating Component for OPE）
    """
    def __init__(self, input_size_full, input_size_task1, input_size_task2,
                 num_specific_experts, num_shared_experts, experts_out, experts_hidden, if_last):
        """
        input_size_full: embedding维度后的全特征输入（用于共享专家）
        input_size_task1: embedding后的任务1选定特征维度
        input_size_task2: embedding后的任务2选定特征维度
        """
        super(CGC_OPE, self).__init__()
        self.if_last = if_last
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        
        # 创建专家：每组专家接收各自对应的 embedding 输入
        self.experts_shared = nn.ModuleList([
            Expert(input_size_full, experts_out, experts_hidden) 
            for _ in range(num_shared_experts)
        ])
        self.experts_task1 = nn.ModuleList([
            Expert(input_size_task1, experts_out, experts_hidden) 
            for _ in range(num_specific_experts)
        ])
        self.experts_task2 = nn.ModuleList([
            Expert(input_size_task2, experts_out, experts_hidden) 
            for _ in range(num_specific_experts)
        ])
        
        # 门控网络分别使用对应的 embedding 作为输入
        self.gate_shared = nn.Sequential(
            nn.Linear(input_size_full, num_specific_experts * 2 + num_shared_experts),
            nn.Softmax(dim=1)
        )
        self.gate_task1 = nn.Sequential(
            nn.Linear(input_size_task1, num_specific_experts + num_shared_experts),
            nn.Softmax(dim=1)
        )
        self.gate_task2 = nn.Sequential(
            nn.Linear(input_size_task2, num_specific_experts + num_shared_experts),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x_full, x_task1, x_task2):
        # x_full: (batch, input_size_full)
        # x_task1: (batch, input_size_task1)
        # x_task2: (batch, input_size_task2)
        batch_size = x_full.size(0)
        
        experts_shared_o = torch.stack([e(x_full) for e in self.experts_shared])  # (num_shared_experts, batch, experts_out)
        experts_task1_o  = torch.stack([e(x_task1) for e in self.experts_task1])    # (num_specific_experts, batch, experts_out)
        experts_task2_o  = torch.stack([e(x_task2) for e in self.experts_task2])    # (num_specific_experts, batch, experts_out)
        
        # 任务1门控
        selected_task1 = self.gate_task1(x_task1)  # (batch, num_specific_experts+num_shared_experts)
        gate_expert_output1 = torch.cat((experts_task1_o, experts_shared_o), dim=0)  # (num_specific_experts+num_shared_experts, batch, experts_out)
        gate_task1_out = torch.einsum('abc, ba -> bc', gate_expert_output1, selected_task1)
        
        # 任务2门控
        selected_task2 = self.gate_task2(x_task2)  # (batch, num_specific_experts+num_shared_experts)
        gate_expert_output2 = torch.cat((experts_task2_o, experts_shared_o), dim=0)
        gate_task2_out = torch.einsum('abc, ba -> bc', gate_expert_output2, selected_task2)
        
        # 共享门控（使用全特征的 embedding）
        selected_shared = self.gate_shared(x_full)  # (batch, num_specific_experts*2+num_shared_experts)
        gate_expert_output_shared = torch.cat((experts_task1_o, experts_task2_o, experts_shared_o), dim=0)
        gate_shared_out = torch.einsum('abc, ba -> bc', gate_expert_output_shared, selected_shared)
        
        if self.if_last:
            return [gate_task1_out, gate_task2_out]
        else:
            return [gate_shared_out, gate_task1_out, gate_task2_out]

class OPE(nn.Module):
    """
    OPE: 任务自适应专家网络，使用门控机制选择不同任务的专家。
    """
    def __init__(self, 
                 input_size_full,  # 原始全特征维度
                 input_size_task1, # 任务1选定特征数量
                 input_size_task2, # 任务2选定特征数量
                 emb_dim_full,     # 全特征embedding后维度
                 emb_dim_task1,    # 任务1选定特征embedding后维度
                 emb_dim_task2,    # 任务2选定特征embedding后维度
                 num_CGC_layers, 
                 num_specific_experts, num_shared_experts,
                 experts_out, experts_hidden, towers_hidden):
        """
        参数说明：
         - input_size_full: 原始全特征维度（如经过预处理后的特征数）
         - input_size_task1: 任务1优选特征数（例如5）
         - input_size_task2: 任务2优选特征数（例如5）
         - emb_dim_full, emb_dim_task1, emb_dim_task2: 对应输入经过 embedding 后的维度
         - 其他参数与原 PLE 模型对应
        """
        super(OPE, self).__init__()
        
        # 分别对原始全特征、任务1选定特征、任务2选定特征进行 embedding
        self.emb_full = nn.Linear(input_size_full, emb_dim_full)
        self.emb_task1 = nn.Linear(input_size_task1, emb_dim_task1)
        self.emb_task2 = nn.Linear(input_size_task2, emb_dim_task2)
        
        # 第一个 CGC_OPE 层
        self.cgc_ope_layer1 = CGC_OPE(
            input_size_full = emb_dim_full,
            input_size_task1 = emb_dim_task1,
            input_size_task2 = emb_dim_task2,
            num_specific_experts = num_specific_experts,
            num_shared_experts = num_shared_experts,
            experts_out = experts_out,
            experts_hidden = experts_hidden,
            if_last = False
        )
        # 其余的 CGC_OPE 层
        self.cgc_ope_layers = nn.ModuleList([
        CGC_OPE(
            input_size_full = experts_out,      # experts_out = 32
            input_size_task1 = experts_out,       # 后续层任务1输入维度也为 experts_out
            input_size_task2 = experts_out,       # 后续层任务2输入维度也为 experts_out
            num_specific_experts = num_specific_experts,
            num_shared_experts = num_shared_experts,
            experts_out = experts_out,
            experts_hidden = experts_hidden,
            if_last = (i == num_CGC_layers - 1)
        )
        for i in range(num_CGC_layers)
    ])
        
        # 任务塔：输入维度为专家输出维度，不再额外拼接选定特征信息（因为该信息已在专家层中使用）
        self.tower1 = Tower(experts_out, 1, towers_hidden)
        self.tower2 = Tower(experts_out, 1, towers_hidden)
        
    def forward(self, x, x1, x2):
        """
        输入：
         - x: 原始全特征输入, shape (batch, input_size_full)
         - x1: 任务1选定特征输入, shape (batch, input_size_task1)
         - x2: 任务2选定特征输入, shape (batch, input_size_task2)
        """
        # 分别获得 embedding
        emb_full = self.emb_full(x)       # (batch, emb_dim_full)
        emb_task1 = self.emb_task1(x1)      # (batch, emb_dim_task1)
        emb_task2 = self.emb_task2(x2)      # (batch, emb_dim_task2)
        
        cgc_out = self.cgc_ope_layer1(emb_full, emb_task1, emb_task2)
        
        for layer in self.cgc_ope_layers:
            cgc_out = layer(cgc_out[0], cgc_out[1], cgc_out[2])
        
        # 任务塔输出
        return [self.tower1(cgc_out[0]), self.tower2(cgc_out[1])]   
