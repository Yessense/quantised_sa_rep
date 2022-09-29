from torch import nn, einsum
import torch
from torch.nn import functional as F
import numpy as np

from .vsa import get_vsa_grid


def get_grid(n):
    x = np.linspace(0, 1.5, n)
    y = np.linspace(0, 1.5, n)
    z = np.linspace(0, 1.5, n)
    grid = np.meshgrid(x, y, z)
    grid = np.stack(grid, axis=-1)
    return grid


def orthgonal_loss_fn(t):
    n = t.shape[0]
    identity = torch.eye(n, device = t.device)
    cosine_sim = einsum('i d, j d -> i j', t, t)
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)


def struct_loss_fn(t, gt_sims):
    n = t.shape[0]
    cosine_sim = einsum('i d, j d -> i j', t, t)
    return ((cosine_sim - gt_sims.to(t.device)) ** 2).sum() / (n ** 2)


def get_distances(t1, t2):
    return (torch.sum(t1**2, dim=1, keepdim=True)
     + torch.sum(t2**2, dim=1) - 2 * torch.matmul(t1, t2.t()))


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if torch.cuda.is_available():
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)
        
        
class CoordQuantizer(nn.Module):
    # def __init__(self, total_embeds=36, embeds_slices=[0, 9, 18, 27, None], nums=[9, 9, 9, 9]):
    def __init__(self, nums=[8, 8, 8, 8]):
        super().__init__()
        self.num_steps = 10
        self.grid1 = nn.Parameter(torch.Tensor(get_grid(self.num_steps))[:, :, :], requires_grad=False) #[4:5, :, :]
        self.linear = nn.Linear(3, 64)
        
        self.num_embeds = [0] + [sum(nums[:i]) for i in range(1, len(nums))] + [None]
        self.nums = nums
        self.total_embeds = sum(nums)

        self.emb_spaces = nn.Parameter(F.normalize(torch.randn(self.total_embeds, 16), p=2, dim=-1))
        # self.emb_spaces = nn.Parameter(F.normalize(torch.randn(2+3+4+9, 16), p=2, dim=-1))
        self.temp = 2.
        self.lins = nn.ModuleList([nn.Linear(64, 16, bias=False) for i in range(len(self.num_embeds) - 1)])
        self.logsmax = nn.LogSoftmax(dim=-1)
        self.len_embeds = len(self.num_embeds) - 1
        self.kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        
    def get_coord_indices(self, inputs, vecs):
        x = inputs
        x = x @ vecs.T
        p_dis = F.softmax(x, dim=-1).mean(dim=[0, 1])

        x = self.logsmax(x)
        log_uniform = torch.log(torch.tensor([1. / self.num_steps**3], device = inputs.device))
        kl_loss = self.kl(log_uniform, x)

        x = x + sample_gumbel(x.size())
        x = F.softmax(x / self.temp, dim=-1)
        samples = [torch.unsqueeze(torch.argmax(x.view(-1, x.shape[2]), dim=-1), dim=-1)]
        return x, kl_loss, p_dis, samples
    
    def use_coord_indices(self, indices, vecs):
        return torch.matmul(indices, vecs)
    
    def get_indices(self, inputs):
        indices = []
        kl_loss = 0.
        p_dis = []
        samples = []
        for lin, i in zip(self.lins, range(self.len_embeds)):
            x = lin(inputs)
            x = x @ self.emb_spaces[self.num_embeds[i]:self.num_embeds[i+1]].T
            # x = x @ self.emb_spaces.T
            
            p_dis.append(F.softmax(x, dim=-1).mean(dim=[0, 1]))
            
            x = self.logsmax(x)
            log_uniform = torch.log(torch.tensor([1. / self.nums[i]], device = inputs.device))
            kl_loss = kl_loss + self.kl(log_uniform, x)

            x = x + sample_gumbel(x.size())
            x = F.softmax(x / self.temp, dim=-1)
            
            samples.append(torch.unsqueeze(torch.argmax(x.view(-1, x.shape[2]), dim=-1), dim=-1))

            indices.append(x)
        return indices, kl_loss, p_dis, samples
    
    def use_indices(self, indices):
        res = []
        for ind, i in zip(indices, range(self.len_embeds)):
            res.append(torch.matmul(ind, self.emb_spaces[self.num_embeds[i]:self.num_embeds[i+1]]))
            # res.append(torch.matmul(ind, self.emb_spaces))
        return torch.cat(res, dim=-1)    
    
    def forward(self, inputs):
        vecs = self.linear(self.grid1).reshape(-1, 64).to(inputs.device)
        
        indices_coord, kl_c, c_dis, c_samples = self.get_coord_indices(inputs, vecs)
        quantized_coord = self.use_coord_indices(indices_coord, vecs)
        
        indices, kl_p, p_dis, p_samples = self.get_indices(inputs)
        quantized = self.use_indices(indices)
        
        loss = (kl_c + kl_p) / 5
        return quantized, quantized_coord, loss
