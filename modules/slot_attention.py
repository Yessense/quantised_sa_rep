from ast import main
import math

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


def norm_prob(mus, logsigmas, values):
    mus = torch.unsqueeze(mus, 2)
    logsigmas = torch.unsqueeze(logsigmas, 2)
    values = torch.unsqueeze(values, 1)
    var = torch.exp(logsigmas)**2
    log_prob =  (-((values - mus) ** 2) / (2 * var)).sum(dim=-1) - logsigmas.sum(dim=-1) - values.shape[-1] * math.log(math.sqrt((2 * math.pi)))
    return torch.exp(log_prob)


class SlotAttentionBase(nn.Module):
    """
    Slot Attention module
    """
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        
    def step(self, slots, k, v, b, n, d, device, n_s):
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=1) + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)

        slots = self.gru(
            updates.reshape(-1, d),
            slots_prev.reshape(-1, d)
        )

        slots = slots.reshape(b, -1, d)
        slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots

    def forward(self, inputs, *args, **kwargs):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots = self.step(slots, k, v, b, n, d, device, n_s)
        #slots = self.step(slots.detach(), k, v, b, n, d, device, n_s)

        return slots
    
    
class SlotAttentionGMM(nn.Module):
    """
    Slot Attention module
    """
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim*2))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim*2))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_q_sig = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru_mu = nn.GRUCell(dim, dim)
        self.gru_logsigma = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)
        self.dim = dim

        self.mlp_mu = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )
        self.mlp_sigma = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim*2)
        self.norm_mu = nn.LayerNorm(dim)
        self.norm_sigma = nn.LayerNorm(dim)
        
        self.mlp_out = nn.Sequential(
            nn.Linear(dim*2, hidden_dim*2),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim*2, dim)
        )
        
    def step(self, slots, k, v, b, n, d, device, n_s, pi_cl):
        slots_prev = slots

        slots = self.norm_slots(slots)
        slots_mu, slots_logsigma = slots.split(self.dim, dim=-1)
        q_mu = self.to_q(slots_mu)
        q_logsigma = self.to_q(slots_logsigma)
        
                
        # probs = norm_prob(slots_mu, slots_logsigma, k) * pi_cl
        # if torch.isnan(probs).any():
        #     print('PROBS Nan appeared')
        # probs = probs / (probs.sum(dim=1, keepdim=True) + self.eps)
        # if torch.isnan(probs).any():
        #     print('PROBS2 Nan appeared')
        # probs = probs / (probs.sum(dim=-1, keepdim=True) + self.eps)
        # if torch.isnan(probs).any():
        #     print('PROBS3 Nan appeared')
        
        #dots = torch.einsum('bid,bjd->bij', q_mu / torch.exp(q_logsigma)**2, k) * self.scale
        dots = ((torch.unsqueeze(k, 1) - torch.unsqueeze(q_mu, 2)) ** 2 / torch.unsqueeze(torch.exp(q_logsigma)**2, 2)).sum(dim=-1) * self.scale
        dots_exp = (torch.exp(-dots) + self.eps) * pi_cl
        attn = dots_exp / dots_exp.sum(dim=1, keepdim=True)
        #attn = (dots.softmax(dim=1) + self.eps)*pi_cl
        #attn = attn / attn.sum(dim=1, keepdim=True)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        
        updates_mu = torch.einsum('bjd,bij->bid', v, attn)
        updates_mu = self.gru_mu(updates_mu.reshape(-1, d), slots_mu.reshape(-1, d))
        updates_mu = updates_mu.reshape(b, -1, d)
        updates_mu = updates_mu + self.mlp_mu(self.norm_mu(updates_mu))
        if torch.isnan(updates_mu).any():
            print('updates_mu Nan appeared')
        
        updates_logsigma = 0.5 * torch.log(torch.einsum('bijd,bij->bid', ((torch.unsqueeze(v, 1) - torch.unsqueeze(updates_mu, 2))**2 + self.eps, attn)))
        #updates_logsigma = updates_logsigma + self.mlp_sigma(self.norm_sigma(updates_logsigma))
        if torch.isnan(updates_logsigma).any():
            print('updates_logsigma Nan appeared')
        slots = torch.cat((updates_mu, updates_logsigma), dim=-1)
        
        pi_cl_new = attn.sum(dim=-1, keepdim=True)
        pi_cl_new = pi_cl_new / (pi_cl_new.sum(dim=1, keepdim=True) + self.eps)

        # updates = torch.einsum('bjd,bij->bid', v, attn)

        # slots = self.gru(
        #     updates.reshape(-1, d*2),
        #     slots_prev.reshape(-1, d*2)
        # )
        if torch.isnan(slots).any():
            print('gru Nan appeared')
        
        if torch.isnan(slots).any():
            print('MLP Nan appeared')
        return slots, pi_cl_new

    def forward(self, inputs, *args, **kwargs):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = self.num_slots
        
        pi_cl = (torch.ones(b, n_s, 1) / n_s).to(device)
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots, pi_cl = self.step(slots, k, v, b, n, d, device, n_s, pi_cl)
        slots, pi_cl = self.step(slots.detach(), k, v, b, n, d, device, n_s, pi_cl)

        return self.mlp_out(slots)

    
class SlotAttention(nn.Module):
    """
    Slot Attention module
    """
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.koefs = nn.Parameter(torch.ones(iters, 2))
        
    def step(self, slots, k, v, b, n, d, device, n_s):
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=1) + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)

        slots = self.gru(
            updates.reshape(-1, d),
            slots_prev.reshape(-1, d)
        )

        slots = slots.reshape(b, -1, d)
        slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots

    def forward(self, inputs, pos, mlp, norm, num_slots = None):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        for i in range(self.iters):
            koefs = F.softmax(self.koefs, dim=-1)
            cur_inputs = inputs * self.koefs[i, 0] + pos * self.koefs[i, 1]
            cur_inputs = mlp(norm(cur_inputs))
            
            cur_inputs = self.norm_input(cur_inputs)      
            k, v = self.to_k(cur_inputs), self.to_v(cur_inputs)
            
            slots = self.step(slots, k, v, b, n, d, device, n_s)
        slots = self.step(slots.detach(), k, v, b, n, d, device, n_s)

        return slots

if __name__ == "__main__":
    slotattention = SlotAttentionBase(num_slots=10, dim=64)
    state_dict = torch.load("/home/alexandr_ko/quantised_sa_od/clevr10_sp")
    key:str
    state_dict = {key[len('slot_attention.'):]:state_dict[key] for key in state_dict if key.startswith('slot_attention')}
    slotattention.load_state_dict(state_dict=state_dict)
    print("DOne")