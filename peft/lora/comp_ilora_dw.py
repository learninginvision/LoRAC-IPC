import torch
import torch.nn as nn
import math
import numpy as np
import torch.distributed as dist

class CompILoraPool(nn.Module):
    def __init__(self, pool_size, depth, dim, rank, 
                 beta1, beta2, lora_alpha=1,
                 lora_qkv=[False, False, True], lora_out=False, lora_fc1=False, lora_fc2=False,):
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
        
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        
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

                # self.register_buffer(attr_name+'_IS', torch.zeros((self.depth, 1, 1)))
                setattr(self, attr_name+'_W', dict())
                self.register_buffer(attr_name+'_IS', torch.zeros((self.pool_size, self.depth,)))
                self.register_buffer(attr_name+'_ISB', torch.zeros((self.pool_size, self.depth,)))
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

                # self.register_buffer(attr_name+'_IS', torch.zeros((self.depth, 1, 1)))
                setattr(self, attr_name+'_W', dict())
                self.register_buffer(attr_name+'_IS', torch.zeros((self.pool_size, self.depth,)))
                self.register_buffer(attr_name+'_ISB', torch.zeros((self.pool_size, self.depth, )))
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

                # self.register_buffer(attr_name+'_IS', torch.zeros((self.depth, 1, 1)))
                setattr(self, attr_name+'_W', dict())
                self.register_buffer(attr_name+'_IS', torch.zeros((self.pool_size, self.depth, )))
                self.register_buffer(attr_name+'_ISB', torch.zeros((self.pool_size, self.depth, )))
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
            for end in ['_A', '_B', '_O', '_OB', '_IS']:
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
            position = ['q', 'k', 'v']
        else:
            position = [position]

        loss = torch.zeros(1, device=device)

        for pos in position:
            if getattr(self, pos+'_lora'):
                lora_A = getattr(self, pos+'_lora_A')
                dim = lora_A.shape[-2]
                lora_A_all = torch.cat([lora_A[:task_id, depth].detach(), lora_A[task_id, depth].unsqueeze(0)], dim=0).permute(0, 2, 1).reshape(-1, dim)
                QQT_A = torch.mm(lora_A_all, lora_A_all.t())
                loss = loss + torch.norm(QQT_A - torch.eye(QQT_A.shape[0], device=device), p='fro')
            
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
                
                self.q_lora_W[depth_id] = self.q_lora_A[task_id, depth_id] @ self.q_lora_B[task_id, depth_id]
                self.k_lora_W[depth_id] = self.k_lora_A[task_id, depth_id] @ self.k_lora_B[task_id, depth_id]
                self.v_lora_W[depth_id] = self.v_lora_A[task_id, depth_id] @ self.v_lora_B[task_id, depth_id]
                
                if self.q_lora_W[depth_id].requires_grad:
                    self.q_lora_W[depth_id].retain_grad()
                if self.k_lora_W[depth_id].requires_grad:
                    self.k_lora_W[depth_id].retain_grad()
                if self.v_lora_W[depth_id].requires_grad:
                    self.v_lora_W[depth_id].retain_grad()

                q = q_his_with_omegas + self.q_lora_W[depth_id]
                k = k_his_with_omegas + self.k_lora_W[depth_id]
                v = v_his_with_omegas + self.v_lora_W[depth_id]

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
                
                self.out_lora_W[depth_id] = self.out_lora_A[task_id, depth_id] @ self.out_lora_B[task_id, depth_id]
                if self.out_lora_W[depth_id].requires_grad  :
                    self.out_lora_W[depth_id].retain_grad()

                w = (out_his_with_omegas + (self.out_lora_W[depth_id])) * self.scaling
            elif position == 'fc1':
                if depth_id >= self.depth:
                    return torch.zeros((self.dim, self.dim * 4), device=device)

                if task_id > 0:
                    with torch.no_grad():
                        fc1_his = torch.einsum('tdr, trl -> tdl', self.fc1_lora_A[:task_id, depth_id, :, :], self.fc1_lora_B[:task_id, depth_id, :, :])
                    fc1_his_with_omegas = (self.fc1_lora_O[:task_id, depth_id].reshape(-1, 1, 1) * fc1_his).sum(dim=0)
                else:
                    fc1_his_with_omegas = 0
                
                self.fc1_lora_W[depth_id] = self.fc1_lora_A[task_id, depth_id] @ self.fc1_lora_B[task_id, depth_id]
                if self.fc1_lora_W[depth_id].requires_grad:
                    self.fc1_lora_W[depth_id].retain_grad()

                w = (fc1_his_with_omegas + (self.fc1_lora_W[depth_id])) * self.scaling
            else:
                if depth_id >= self.depth:
                    return torch.zeros((self.dim * 4, self.dim), device=device)

                if task_id > 0:
                    with torch.no_grad():
                        fc2_his = torch.einsum('tdr, trl -> tdl', self.fc2_lora_A[:task_id, depth_id, :, :], self.fc2_lora_B[:task_id, depth_id, :, :])
                    fc2_his_with_omegas = (self.fc2_lora_O[:task_id, depth_id].reshape(-1, 1, 1) * fc2_his).sum(dim=0)
                else:
                    fc2_his_with_omegas = 0
                
                self.fc2_lora_W[depth_id] = self.fc2_lora_A[task_id, depth_id] @ self.fc2_lora_B[task_id, depth_id]
                if self.fc2_lora_W[depth_id].requires_grad:
                    self.fc2_lora_W[depth_id].retain_grad()

                w = (fc2_his_with_omegas + (self.fc2_lora_W[depth_id])) * self.scaling    
            
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


    def loss_reg(self, task_id=-1, device=None, args=None, dtype='l2'):
        loss = torch.zeros(1, device=device)

        if task_id > 0:
            for depth in range(self.depth):
                for position in ['qkv', 'out', 'fc1', 'fc2']:
                    position = position.split('_')[0]
                    current_delta_w = self.cal_delta_w(device, task_id, depth, True, position)
                    previous_delta_w = self.cal_delta_w(device, task_id-1, depth, False, position)
                    diff_delta_w = current_delta_w - previous_delta_w
                    # lora_A = getattr(self, attr_name+'_A') # (pool_size, depth, dim, r)
                    # lora_B = getattr(self, attr_name+'_B') # (pool_size, depth, r, dim)
                    # lora_O = getattr(self, attr_name+'_O') # (pool_size, depth)
                    # current_delta_W = torch.eninsum('dhr, drl -> dhl', lora_A[task_id], lora_B[task_id]) * lora_O[task_id].reshape(-1, 1, 1) / self.scaling               
                    if dtype == 'l2':
                        if position == 'qkv':
                            diff_delta_w_list = diff_delta_w.chunk(3, dim=-1)
                            for idx, pos in enumerate(list(position)):
                                lora_IS = getattr(self, f'{pos}_lora_IS')
                                loss = loss + torch.mean(diff_delta_w_list[idx] ** 2 * lora_IS[task_id-1][depth])
                        else:
                            lora_IS = getattr(self, position+'_lora_IS')
                            loss = loss + torch.mean(diff_delta_w ** 2 * lora_IS[task_id-1][depth])
                    elif dtype == 'l2_norm':
                        if position == 'qkv':
                            diff_delta_w_list = diff_delta_w.chunk(3, dim=-1)
                            for idx, pos in enumerate(list(position)):
                                lora_IS = getattr(self, f'{pos}_lora_IS')
                                loss = loss + lora_IS[task_id-1][depth] * torch.norm(diff_delta_w_list[idx], p=2) ** 2
                        else:
                            lora_IS = getattr(self, position+'_lora_IS')
                            loss = loss + lora_IS[task_id-1][depth] * torch.norm(diff_delta_w, p=2) ** 2
                    else:
                        raise NotImplementedError
            
        return loss

    def forward(self, x, task_id=-1, depth_id=-1, train=False, position='qkv'):
        ret = dict()

        w = self.cal_delta_w(x.device, task_id, depth_id, train, position)

        ret['lora_value'] = torch.einsum('bld, dz->blz', x, w) # B x L x 3dim

        return ret

    def calculate_score(self, name, task_id=None, metric="ipt"):
        if metric == "ipt":
            ipt_score = self.exp_avg_ipt[name] * self.exp_avg_unc[name]
        else:
            raise ValueError("Unexcptected Metric: %s"%metric)
        return ipt_score

    def update_ipt(self, task_id): 
        for attr_name in self.attributes:
            cond = getattr(self, attr_name)
            if cond:
                for end in ['_W']:
                    name = attr_name + end
                    param = getattr(self, name)
                    if name not in self.ipt:
                        self.ipt[name] = torch.zeros(self.depth)
                        self.exp_avg_ipt[name] = torch.zeros(self.depth)
                        self.exp_avg_unc[name] = torch.zeros(self.depth)
                        
                    for i in range(self.depth):
                        with torch.no_grad():                     
                            self.ipt[name][i] = (param[i]  * param[i].grad).abs().detach().mean()
                            # Update sensitivity 
                            self.exp_avg_ipt[name][i] = self.beta1 * self.exp_avg_ipt[name][i] + (1 - self.beta1) * self.ipt[name][i] 
                            # Update uncertainty 
                            self.exp_avg_unc[name][i] = self.beta2 * self.exp_avg_unc[name][i] + (1 - self.beta2) * (self.ipt[name][i]   - self.exp_avg_ipt[name][i]  ).abs()
    
    
    @torch.no_grad()
    def balance_ipt(self, task_id, device, world_size):
        for attr_name in self.attributes:
            cond = getattr(self, attr_name)
            if cond:
                for end in ['_W']:
                    name = attr_name + end
                    param = getattr(self, name)

                    for i in range(self.depth):
                        grad = param[i].grad
                        grad_list = [torch.zeros_like(grad, device=device) for _ in range(world_size)]
                        dist.barrier()
                        dist.all_gather(grad_list, grad)
                        
                        grad = torch.stack(grad_list, dim=0).mean(dim=0)
                        param[i].grad = grad

                            
    @torch.no_grad()
    def after_task(self, task_id, device=None):
        assert isinstance(task_id, int)
        if task_id < self.pool_size:
            # attributes = ['q_lora', 'k_lora', 'v_lora', 'out_lora', 'fc1_lora', 'fc2_lora']
            for attr_name in self.attributes:
                omegas_bank_buffer = getattr(self, attr_name+'_OB')
                omegas_param = getattr(self, attr_name+'_O')
                omegas_bank_buffer[task_id] = omegas_param.clone().detach()
                
                cond = getattr(self, attr_name)
                if cond:
                    lora_ipt_score = self.calculate_score(name=attr_name+'_W', task_id=task_id, metric="ipt") #.mean(dim=(1, 2), keepdim=True)
                    
                    lora_IS = getattr(self, attr_name+'_IS')
                    lora_IS[task_id] = lora_ipt_score
                    lora_IS_bank = getattr(self, attr_name+'_ISB')
                    lora_IS_bank[task_id] = lora_ipt_score.clone().detach()
        
        for attr_name in self.attributes:
            cond = getattr(self, attr_name)
            if cond:
                for end in ['_W']:
                    name = attr_name + end
                    self.ipt[name] = torch.zeros(self.depth)
                    self.exp_avg_ipt[name] = torch.zeros(self.depth)
                    self.exp_avg_unc[name] = torch.zeros(self.depth)

        self.current_task = self.current_task + 1
        assert int(self.current_task.cpu()) == task_id + 1, 'current task is {}, but task_id is {}'.format(self.current_task, task_id)
        
    def get_lora_ipt(self):
        lora_ipt = {}
        for attr_name in self.attributes:
            cond = getattr(self, attr_name)
            if cond:
                lora_ipt[attr_name] = getattr(self, attr_name+'_ISB').cpu().numpy()
        
        return lora_ipt
    
    def cal_diff_delta_w(self, task_id=-1, device=None, args=None,):
        diff_delta_ws = {}
        for position in ['qkv', 'out', 'fc1', 'fc2']:
            if position == 'qkv':
                for idx, pos in enumerate(list(position)):
                    diff_delta_ws[pos] = []
            else:
                diff_delta_ws[position] = []
        
        for depth in range(self.depth):
            for position in ['qkv', 'out', 'fc1', 'fc2']:
                position = position.split('_')[0]
                current_delta_w = self.cal_delta_w(device, task_id, depth, False, position)
                previous_delta_w = self.cal_delta_w(device, task_id-1, depth, False, position)
                diff_delta_w = current_delta_w - previous_delta_w
                
                if position == 'qkv':
                    diff_delta_w_list = diff_delta_w.chunk(3, dim=-1)
                    for idx, pos in enumerate(list(position)):
                        diff_delta_w_norm = torch.norm(diff_delta_w_list[idx], p=2) ** 2
                        # diff_delta_w_norm = torch.sum(torch.abs(diff_delta_w_list[idx]))
                        # diff_delta_w_norm = torch.mean(diff_delta_w_list[idx] ** 2)
                        diff_delta_ws[pos].append(diff_delta_w_norm)
                else:
                    diff_delta_w_norm = torch.norm(diff_delta_w, p=2) ** 2
                    # diff_delta_w_norm = torch.sum(torch.abs(diff_delta_w))
                    # diff_delta_w_norm = torch.mean(diff_delta_w ** 2)
                    diff_delta_ws[position].append(diff_delta_w_norm)
                    
        return diff_delta_ws
    
    def get_omegas(self):
        omegas = {}
        for attr_name in self.attributes:
            cond = getattr(self, attr_name)
            if cond:
                omegas[attr_name] = getattr(self, attr_name+'_O').data
        
        return omegas