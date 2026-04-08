
import torch

def calc_cc_sim_batch(pred, gt):

    p = pred.view(-1, pred.size(-2) * pred.size(-1)) 
    g = gt.view(-1, gt.size(-2) * gt.size(-1))
    

    p_mean = p.mean(dim=1, keepdim=True)
    p_std = p.std(dim=1, keepdim=True)
    g_mean = g.mean(dim=1, keepdim=True)
    g_std = g.std(dim=1, keepdim=True)
    
    a = (p - p_mean) / (p_std + 1e-7)
    b = (g - g_mean) / (g_std + 1e-7)
    
    cc = (a * b).sum(dim=1) / torch.sqrt((a*a).sum(dim=1) * (b*b).sum(dim=1) + 1e-7)
    
    p_norm = p / (p.sum(dim=1, keepdim=True) + 1e-7)
    g_norm = g / (g.sum(dim=1, keepdim=True) + 1e-7)
    
    sim = torch.sum(torch.minimum(p_norm, g_norm), dim=1)
    
    return cc.mean().item(), sim.mean().item()