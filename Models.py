import numpy as np
import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, input_feature_size, expert_hidden_size, expert_output_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_feature_size, expert_hidden_size)
        self.fc2 = nn.Linear(expert_hidden_size, expert_output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class CGC(nn.Module):
    def __init__(self, input_size, num_specific_experts, num_shared_experts, experts_out, experts_hidden, if_last):
        super(CGC, self).__init__()

        self.input_size = input_size  # 输入特征维度
        self.num_specific_experts = num_specific_experts  # 每个任务的专家数量
        self.num_shared_experts = num_shared_experts  # 共享专家数量
        self.experts_out = experts_out  # 专家输出维度
        self.experts_hidden = experts_hidden  # 专家隐藏层维度
        self.if_last = if_last

        self.experts_shared = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_shared_experts)])  # 共享专家的组合
        self.experts_task1 = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_specific_experts)])  # 任务1的专家组合
        self.experts_task2 = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_specific_experts)])  # 任务2的专家组合

        self.soft = nn.Softmax(dim=1)

        self.gate_shared = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts*2 + self.num_shared_experts), nn.Softmax(dim=1))  # 共享专家门控网络（为什么输出维度是所有相加？）
        self.gate_task1 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts), nn.Softmax(dim=1))  # 任务1专家门控网络
        self.gate_task2 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts), nn.Softmax(dim=1))  # 任务2专家门控网络

    def forward(self, x):
        inputs_shared, inputs_task1, inputs_task2 = x
 
        experts_shared_o = [e(inputs_shared) for e in self.experts_shared]
        experts_shared_o = torch.stack(experts_shared_o)  # num_shared_experts * batch_size * experts_out
        
        experts_task1_o = [e(inputs_task1) for e in self.experts_task1]
        experts_task1_o = torch.stack(experts_task1_o)  # num_specific_experts * batch_size * experts_out
        
        experts_task2_o = [e(inputs_task2) for e in self.experts_task2]
        experts_task2_o = torch.stack(experts_task2_o)  # num_specific_experts * batch_size * experts_out

        #gate1
        selected_task1 = self.gate_task1(inputs_task1)  # batch_size * (num_specific_experts + num_shared_experts)
        gate_expert_output1 = torch.cat((experts_task1_o, experts_shared_o), dim=0)  # (num_specific_experts + num_shared_experts) * batch_size * experts_out
        # 利用爱因斯坦求和（einsum）进行加权求和
        gate_task1_out = torch.einsum('abc, ba -> bc', gate_expert_output1, selected_task1)  # batch_size * experts_out

        #gate2
        selected_task2 = self.gate_task2(inputs_task2)
        gate_expert_output2 = torch.cat((experts_task2_o, experts_shared_o), dim=0)
        gate_task2_out = torch.einsum('abc, ba -> bc', gate_expert_output2, selected_task2)  # batch_size * experts_out

        #gate_shared
        selected_shared = self.gate_shared(inputs_shared)  # batch_size * (num_specific_experts*2 + num_shared_experts)
        gate_expert_outputshared = torch.cat((experts_task1_o, experts_task2_o, experts_shared_o), dim=0)  # (num_specific_experts*2 + num_shared_experts) * batch_size * experts_out
        gate_shared_out = torch.einsum('abc, ba -> bc', gate_expert_outputshared, selected_shared)  # batch_size * experts_out

        if self.if_last:
            return [gate_task1_out, gate_task2_out]
        else:
            return [gate_shared_out, gate_task1_out, gate_task2_out]

class PLE(nn.Module):
    def __init__(self, num_CGC_layers, input_size, num_specific_experts, num_shared_experts, experts_out,
                 experts_hidden, towers_hidden):
        super(PLE, self).__init__()
        self.num_CGC_layers = num_CGC_layers
        self.input_size = input_size
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        
        self.cgc_layer1 = CGC(self.input_size, self.num_specific_experts, self.num_shared_experts, self.experts_out, self.experts_hidden, if_last=False)
        self.cgc_layers = nn.ModuleList([CGC(32, num_specific_experts, num_shared_experts, experts_out, experts_hidden, if_last=(i == num_CGC_layers - 1)) for i in range(num_CGC_layers)])

        self.tower1 = Tower(self.experts_out, 1, self.towers_hidden)
        self.tower2 = Tower(self.experts_out, 1, self.towers_hidden)

    def forward(self, x):
        cgc_outputs = self.cgc_layer1([x , x, x])
        

        for cgc_layer in self.cgc_layers:
            cgc_outputs = cgc_layer(cgc_outputs)
            
        final_output1 = self.tower1(cgc_outputs[0])
        final_output2 = self.tower2(cgc_outputs[1])

        return [final_output1, final_output2]


# In[ ]:




