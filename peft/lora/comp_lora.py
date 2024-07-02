import torch
import torch.nn as nn
import math
import numpy as np

class CompLoraPool(nn.Module):
    def __init__(self, pool_size, depth, dim, rank, lora_alpha=1, lora_qkv=[False, False, True], lora_out=False, lora_fc1=False, lora_fc2=False):
        super().__init__()
        self.r = rank
        self.lora_alpha = lora_alpha
        self.scaling = torch.tensor(self.lora_alpha / self.r)
        self.qkv_lora = lora_qkv
        self.q_lora, self.k_lora, self.v_lora = self.qkv_lora[0], self.qkv_lora[1], self.qkv_lora[2]
        self.out_lora = lora_out
        self.fc1_lora = lora_fc1
        self.fc2_lora = lora_fc2
        assert isinstance(lora_qkv, list) and len(self.qkv_lora) == 3
        self.depth = depth
        self.pool_size = pool_size
        self.dim = dim
        self.register_buffer('current_task', torch.zeros(1, dtype=torch.int32))
        self.attributes = ('q_lora', 'k_lora', 'v_lora', 'out_lora', 'fc1_lora', 'fc2_lora')
        self.position = ('qkv', 'out', 'fc1', 'fc2')
        self.create_parameters()
        self.reset_parameters()
        assert self.q_lora or self.k_lora or self.v_lora or self.out_lora or self.fc1_lora or self.fc2_lora

    def create_parameters(self):
        # attributes = ['q_lora', 'k_lora', 'v_lora', 'out_lora', 'fc1_lora', 'fc2_lora']
        for attr_name in self.attributes:
            cond = getattr(self, attr_name)
            if attr_name in ['q_lora', 'k_lora', 'v_lora', 'out_lora']:
                if cond:
                    setattr(self, attr_name+'_A', nn.Parameter(torch.zeros((self.pool_size, self.depth, self.dim, self.r))))
                    setattr(self, attr_name+'_B', nn.Parameter(torch.zeros((self.pool_size, self.depth, self.r, self.dim))))
                    setattr(self, attr_name+'_O', nn.Parameter(torch.ones((self.pool_size, self.depth))))
                else:
                    setattr(self, attr_name+'_A', torch.zeros((self.pool_size, self.depth, self.dim, self.r)))
                    setattr(self, attr_name+'_B', torch.zeros((self.pool_size, self.depth, self.r, self.dim)))
                    setattr(self, attr_name+'_O', torch.ones((self.pool_size, self.depth)))

                self.register_buffer(attr_name+'_OB', torch.zeros((self.pool_size, self.pool_size, self.depth)))

            elif attr_name == 'fc1_lora':
                if cond:
                    setattr(self, attr_name+'_A', nn.Parameter(torch.zeros((self.pool_size, self.depth, self.dim, self.r))))
                    setattr(self, attr_name+'_B', nn.Parameter(torch.zeros((self.pool_size, self.depth, self.r, self.dim * 4))))
                    setattr(self, attr_name+'_O', nn.Parameter(torch.ones((self.pool_size, self.depth))))
                else:
                    setattr(self, attr_name+'_A', torch.zeros((self.pool_size, self.depth, self.dim, self.r)))
                    setattr(self, attr_name+'_B', torch.zeros((self.pool_size, self.depth, self.r, self.dim * 4)))
                    setattr(self, attr_name+'_O', torch.ones((self.pool_size, self.depth)))

                self.register_buffer(attr_name+'_OB', torch.zeros((self.pool_size, self.pool_size, self.depth)))

            elif attr_name == 'fc2_lora':
                if cond:
                    setattr(self, attr_name+'_A', nn.Parameter(torch.zeros((self.pool_size, self.depth, self.dim * 4, self.r))))
                    setattr(self, attr_name+'_B', nn.Parameter(torch.zeros((self.pool_size, self.depth, self.r, self.dim))))
                    setattr(self, attr_name+'_O', nn.Parameter(torch.ones((self.pool_size, self.depth))))
                else:
                    setattr(self, attr_name+'_A', torch.zeros((self.pool_size, self.depth, self.dim * 4, self.r)))
                    setattr(self, attr_name+'_B', torch.zeros((self.pool_size, self.depth, self.r, self.dim)))
                    setattr(self, attr_name+'_O', torch.ones((self.pool_size, self.depth)))

                self.register_buffer(attr_name+'_OB', torch.zeros((self.pool_size, self.pool_size, self.depth)))
            else:
                raise NotImplementedError
            
    def reset_parameters(self):
        # attributes = ['q_lora', 'k_lora', 'v_lora', 'out_lora', 'fc1_lora', 'fc2_lora']
        for attr_name in self.attributes:
            for end in ['_A', '_B', '_O']:
                param = getattr(self, attr_name+end)
                if isinstance(param, nn.Parameter):
                    if end == '_A':
                        p, d, _, _ = param.shape
                        for i in range(p):
                            for j in range(d):
                                nn.init.kaiming_uniform_(param[i][j], a=math.sqrt(5))
                    elif end == '_O':
                        nn.init.ones_(param)
                    else:
                        nn.init.zeros_(param)
        
    def to_device(self, device):
        # attributes = ['q_lora', 'k_lora', 'v_lora', 'out_lora', 'fc1_lora', 'fc2_lora']
        for attr_name in self.attributes:
            for end in ['_A', '_B', '_O', '_OB']:
                param = getattr(self, attr_name+end)
                if param is None:
                    print(attr_name+end)
                
                if not isinstance(param, nn.Parameter):
                    setattr(self, attr_name+end, param.to(device))

    def loss_ortho(self, task_id=None, device=None, depth=None, position=None, args=None):
        assert position in ('qkv', 'out', 'fc1', 'fc2')
        assert isinstance(task_id, int)

        if depth >= self.depth:
            return torch.zeros(1, device=device)

        if position == 'qkv':
            if self.q_lora:
                q_lora_A_all = torch.cat([self.q_lora_A[:task_id, depth].detach(), self.q_lora_A[task_id, depth].unsqueeze(0)], dim=0).permute(0, 2, 1).reshape(-1, self.dim)
                QQT = torch.mm(q_lora_A_all, q_lora_A_all.t())
                loss_q = torch.norm(QQT - torch.eye(QQT.shape[0], device=device), p='fro')
            else:
                loss_q = torch.zeros(1, device=device)

            if self.k_lora:
                k_lora_A_all = torch.cat([self.k_lora_A[:task_id, depth].detach(), self.k_lora_A[task_id, depth].unsqueeze(0)], dim=0).permute(0, 2, 1).reshape(-1, self.dim)
                KKT = torch.mm(k_lora_A_all, k_lora_A_all.t())
                loss_k = torch.norm(KKT - torch.eye(KKT.shape[0], device=device), p='fro')
            else:
                loss_k = torch.zeros(1, device=device)
            
            if self.v_lora:
                v_lora_A_all = torch.cat([self.v_lora_A[:task_id, depth].detach(), self.v_lora_A[task_id, depth].unsqueeze(0)], dim=0).permute(0, 2, 1).reshape(-1, self.dim)
                VVT = torch.mm(v_lora_A_all, v_lora_A_all.t())
                loss_v = torch.norm(VVT - torch.eye(VVT.shape[0], device=device), p='fro')
            else:
                loss_v = torch.zeros(1, device=device)

            loss = loss_q + loss_k + loss_v
        elif position == 'out':
            if self.out_lora:
                out_lora_A_all = torch.cat([self.out_lora_A[:task_id, depth].detach(), self.out_lora_A[task_id, depth].unsqueeze(0)], dim=0).permute(0, 2, 1).reshape(-1, self.dim)
                OOT = torch.mm(out_lora_A_all, out_lora_A_all.t())
                loss = torch.norm(OOT - torch.eye(OOT.shape[0], device=device), p='fro')
            else:
                loss = torch.zeros(1, device=device)
        elif position == 'fc1':
            if self.fc1_lora:
                fc1_lora_A_all = torch.cat([self.fc1_lora_A[:task_id, depth].detach(), self.fc1_lora_A[task_id, depth].unsqueeze(0)], dim=0).permute(0, 2, 1).reshape(-1, self.dim)
                FCT = torch.mm(fc1_lora_A_all, fc1_lora_A_all.t())
                loss = torch.norm(FCT - torch.eye(FCT.shape[0], device=device), p='fro')
            else:
                loss = torch.zeros(1, device=device)
        else:
            if self.fc2_lora:
                fc2_lora_A_all = torch.cat([self.fc2_lora_A[:task_id, depth].detach(), self.fc2_lora_A[task_id, depth].unsqueeze(0)], dim=0).permute(0, 2, 1).reshape(-1, self.dim * 4)
                FCT = torch.mm(fc2_lora_A_all, fc2_lora_A_all.t())
                loss = torch.norm(FCT - torch.eye(FCT.shape[0], device=device), p='fro')
            else:
                loss = torch.zeros(1, device=device)
        
        return loss

    def cal_delta_w(self, device, task_id=-1, depth_id=-1, train=False, position='qkv'):
        self.to_device(device)
        assert position in ('qkv', 'out', 'fc1', 'fc2')
        if train:
            assert isinstance(task_id, int)
            if position == 'qkv':
                if depth_id >= self.depth:
                    return torch.zeros((self.dim, self.dim * 3), device=device)

                if task_id > 0:
                    with torch.no_grad():
                        q_his = torch.einsum('tdr, trl -> tdl', self.q_lora_A[:task_id, depth_id, :, :], self.q_lora_B[:task_id, depth_id, :, :])
                        k_his = torch.einsum('tdr, trl -> tdl', self.k_lora_A[:task_id, depth_id, :, :], self.k_lora_B[:task_id, depth_id, :, :])
                        v_his = torch.einsum('tdr, trl -> tdl', self.v_lora_A[:task_id, depth_id, :, :], self.v_lora_B[:task_id, depth_id, :, :])

                    q_his_with_omegas = (self.q_lora_O[:task_id, depth_id].reshape(-1, 1, 1) * q_his).sum(dim=0)
                    k_his_with_omegas = (self.k_lora_O[:task_id, depth_id].reshape(-1, 1, 1) * k_his).sum(dim=0)
                    v_his_with_omegas = (self.v_lora_O[:task_id, depth_id].reshape(-1, 1, 1) * v_his).sum(dim=0)
                else:
                    q_his_with_omegas = 0
                    k_his_with_omegas = 0
                    v_his_with_omegas = 0

                q = q_his_with_omegas + (self.q_lora_A[task_id, depth_id] @ self.q_lora_B[task_id, depth_id])
                k = k_his_with_omegas + (self.k_lora_A[task_id, depth_id] @ self.k_lora_B[task_id, depth_id])
                v = v_his_with_omegas + (self.v_lora_A[task_id, depth_id] @ self.v_lora_B[task_id, depth_id])

                w = torch.cat([q.to(device), k.to(device), v.to(device)], dim=-1) * self.scaling
            elif position == 'out':
                if depth_id >= self.depth:
                    return torch.zeros((self.dim, self.dim), device=device)

                if task_id > 0:
                    with torch.no_grad():
                        out_his = torch.einsum('tdr, trl -> tdl', self.out_lora_A[:task_id, depth_id, :, :], self.out_lora_B[:task_id, depth_id, :, :])
                    
                    out_his_with_omegas = (self.out_lora_O[:task_id, depth_id].reshape(-1, 1, 1) * out_his).sum(dim=0)
                else:
                    out_his_with_omegas = 0

                w = (out_his_with_omegas + (self.out_lora_A[task_id, depth_id] @ self.out_lora_B[task_id, depth_id])) * self.scaling
            elif position == 'fc1':
                if depth_id >= self.depth:
                    return torch.zeros((self.dim, self.dim * 4), device=device)

                if task_id > 0:
                    with torch.no_grad():
                        fc1_his = torch.einsum('tdr, trl -> tdl', self.fc1_lora_A[:task_id, depth_id, :, :], self.fc1_lora_B[:task_id, depth_id, :, :])
                    fc1_his_with_omegas = (self.fc1_lora_O[:task_id, depth_id].reshape(-1, 1, 1) * fc1_his).sum(dim=0)
                else:
                    fc1_his_with_omegas = 0

                w = (fc1_his_with_omegas + (self.fc1_lora_A[task_id, depth_id] @ self.fc1_lora_B[task_id, depth_id])) * self.scaling
            else:
                if depth_id >= self.depth:
                    return torch.zeros((self.dim * 4, self.dim), device=device)

                if task_id > 0:
                    with torch.no_grad():
                        fc2_his = torch.einsum('tdr, trl -> tdl', self.fc2_lora_A[:task_id, depth_id, :, :], self.fc2_lora_B[:task_id, depth_id, :, :])
                    fc2_his_with_omegas = (self.fc2_lora_O[:task_id, depth_id].reshape(-1, 1, 1) * fc2_his).sum(dim=0)
                else:
                    fc2_his_with_omegas = 0

                w = (fc2_his_with_omegas + (self.fc2_lora_A[task_id, depth_id] @ self.fc2_lora_B[task_id, depth_id])) * self.scaling    
            
        else:
            assert task_id < int(self.current_task.cpu()), 'current task is {}, but task_id is {}'.format(int(self.current_task.cpu()), task_id)
            assert isinstance(task_id, int)
            if position == 'qkv':
                if depth_id >= self.depth:
                    return torch.zeros((self.dim, self.dim * 3), device=device)

                with torch.no_grad():
                    q = torch.einsum('tdr, trl -> tdl', self.q_lora_A[:task_id+1, depth_id, :, :], self.q_lora_B[:task_id+1, depth_id, :, :])
                    k = torch.einsum('tdr, trl -> tdl', self.k_lora_A[:task_id+1, depth_id, :, :], self.k_lora_B[:task_id+1, depth_id, :, :])
                    v = torch.einsum('tdr, trl -> tdl', self.v_lora_A[:task_id+1, depth_id, :, :], self.v_lora_B[:task_id+1, depth_id, :, :])
                    q_with_omegas = (self.q_lora_OB[task_id, :task_id+1, depth_id].reshape(-1, 1, 1) * q).sum(dim=0)
                    k_with_omegas = (self.k_lora_OB[task_id, :task_id+1, depth_id].reshape(-1, 1, 1) * k).sum(dim=0)
                    v_with_omegas = (self.v_lora_OB[task_id, :task_id+1, depth_id].reshape(-1, 1, 1) * v).sum(dim=0)

                w = torch.cat([q_with_omegas.to(device), k_with_omegas.to(device), v_with_omegas.to(device)], dim=-1) * self.scaling
            elif position == 'out':
                if depth_id >= self.depth:
                    return torch.zeros((self.dim, self.dim), device=device)
                
                with torch.no_grad():
                    out = torch.einsum('tdr, trl -> tdl', self.out_lora_A[:task_id+1, depth_id, :, :], self.out_lora_B[:task_id+1, depth_id, :, :])
                    out_with_omegas = (self.out_lora_OB[task_id, :task_id+1, depth_id].reshape(-1, 1, 1) * out).sum(dim=0)

                w = out_with_omegas * self.scaling
            elif position == 'fc1':
                if depth_id >= self.depth:
                    return torch.zeros((self.dim, self.dim * 4), device=device)
                
                with torch.no_grad():
                    fc1 = torch.einsum('tdr, trl -> tdl', self.fc1_lora_A[:task_id+1, depth_id, :, :], self.fc1_lora_B[:task_id+1, depth_id, :, :])
                    fc1_with_omegas = (self.fc1_lora_OB[task_id, :task_id+1, depth_id].reshape(-1, 1, 1) * fc1).sum(dim=0)

                w = fc1_with_omegas * self.scaling
            else:
                if depth_id >= self.depth:
                    return torch.zeros((self.dim * 4, self.dim), device=device)
                
                with torch.no_grad():
                    fc2 = torch.einsum('tdr, trl -> tdl', self.fc2_lora_A[:task_id+1, depth_id, :, :], self.fc2_lora_B[:task_id+1, depth_id, :, :])
                    fc2_with_omegas = (self.fc2_lora_OB[task_id, :task_id+1, depth_id].reshape(-1, 1, 1) * fc2).sum(dim=0)

                w = fc2_with_omegas * self.scaling
        
        return w

    def loss_similarity(self, task_id=-1, device=None, args=None, dtype='cosine'):
    
        if task_id > 0:
            loss = torch.zeros(1, device=device)

            for d in range(self.depth):
                for attr_name in self.attributes:
                    cond = getattr(self, attr_name)
                    if cond:
                        param_A = getattr(self, attr_name+'_A')
                        param_B = getattr(self, attr_name+'_B')
                        current_delta_W = param_A[task_id, d] @ param_B[task_id, d]
                        if dtype == 'l2':
                            loss = loss + torch.mean(current_delta_W ** 2)
                        elif dtype == 'l1':
                            loss = loss + torch.mean(torch.abs(current_delta_W))

            return loss
        else:
            return torch.tensor(0.0, device=device)


    def forward(self, x, task_id=-1, depth_id=-1, train=False, position='qkv'):
        ret = dict()

        w = self.cal_delta_w(x.device, task_id, depth_id, train, position)

        ret['lora_value'] = torch.einsum('bld, dz->blz', x, w) # B x L x 3dim

        return ret
    
    
    @torch.no_grad()
    def after_task(self, task_id, device=None):
        assert isinstance(task_id, int)
        if task_id < self.pool_size:
            # attributes = ['q_lora', 'k_lora', 'v_lora', 'out_lora', 'fc1_lora', 'fc2_lora']
            for attr_name in self.attributes:
                omegas_bank_buffer = getattr(self, attr_name+'_OB')
                omegas_param = getattr(self, attr_name+'_O')
                omegas_bank_buffer[task_id] = omegas_param.clone().detach()

        self.current_task = self.current_task + 1
        assert int(self.current_task.cpu()) == task_id + 1, 'current task is {}, but task_id is {}'.format(self.current_task, task_id)