
import torch
import torch.nn as nn
import torch.nn.functional as F

def torch_nss(pred, fixation_mask):
    # pred, fixation_mask: (B, 1, T, H, W)
    B, C, T, H, W = pred.shape
    
    p = pred.view(B * T, -1)
    f = fixation_mask.view(B * T, -1)

    p_mean = p.mean(dim=1, keepdim=True)
    p_std = p.std(dim=1, keepdim=True)
    p_norm = (p - p_mean) / (p_std + 1e-7)

    num_fixations = f.sum(dim=1) + 1e-7
    nss_per_frame = torch.sum(p_norm * f, dim=1) / num_fixations
    
    valid_mask = f.sum(dim=1) > 0
    if valid_mask.any():
        return nss_per_frame[valid_mask].mean()
    return torch.tensor(0.0, device=pred.device)

class VSPLoss(nn.Module):
    def __init__(self, kl_weight=10.0, cc_weight=2.0, sim_weight=1.0, nss_weight=0, bce_weight=1.0):

        super().__init__()
        self.kl_weight = kl_weight
        self.cc_weight = cc_weight
        self.sim_weight = sim_weight
        self.nss_weight = nss_weight
        self.bce_weight = bce_weight
        # self.bce = nn.BCELoss()
        self.fixation_alpha = 3.0

    def forward(self, pred, target, fixations=None):

        B, C, T, H, W = pred.shape
        
        pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
        target = torch.clamp(target, min=0.0, max=1.0)

        p_frames = pred.transpose(1, 2).reshape(B * T, -1)
        g_frames = target.transpose(1, 2).reshape(B * T, -1)

        p_norm = p_frames / (p_frames.sum(dim=1, keepdim=True) + 1e-7)
        g_norm = g_frames / (g_frames.sum(dim=1, keepdim=True) + 1e-7)

        kl = torch.sum(g_norm * torch.log(g_norm / (p_norm + 1e-7) + 1e-7), dim=1)
        loss_kl = kl.mean()

        p_centered = p_frames - p_frames.mean(dim=1, keepdim=True)
        g_centered = g_frames - g_frames.mean(dim=1, keepdim=True)
        cc_num = (p_centered * g_centered).sum(dim=1)
        cc_den = torch.sqrt((p_centered**2).sum(dim=1) * (g_centered**2).sum(dim=1) + 1e-7)
        cc = cc_num / cc_den
        loss_cc = (1.0 - cc).mean()

        sim = torch.sum(torch.minimum(p_norm, g_norm), dim=1)
        loss_sim = (1.0 - sim).mean()

        bce_raw = F.binary_cross_entropy(pred, target, reduction='none')

        if fixations is not None:

            weight_map = 1.0 + (self.fixation_alpha * fixations)

            loss_bce = (bce_raw * weight_map).mean()
        else:

            loss_bce = bce_raw.mean()
        
        loss_nss = torch.tensor(0.0, device=pred.device)
        if fixations is not None:

            p_flat = pred.transpose(1, 2).reshape(B * T, -1)
            f_flat = fixations.transpose(1, 2).reshape(B * T, -1)
            g_flat = target.transpose(1, 2).reshape(B * T, -1) 

            weighted_f = f_flat * g_flat.detach() 
            
            p_mean = p_flat.mean(dim=1, keepdim=True).detach()
            p_std = p_flat.std(dim=1, keepdim=True).detach()
            p_zscore = (p_flat - p_mean) / (p_std + 1e-7)
            
            num_fix = weighted_f.sum(dim=1) + 1e-7
            nss_scores = torch.sum(p_zscore * weighted_f, dim=1) / num_fix
            
            valid_frames = (f_flat.sum(dim=1) > 0.5)
            if valid_frames.any():
                loss_nss = -nss_scores[valid_frames].mean()

        total_loss = (self.kl_weight * loss_kl + 
                      self.cc_weight * loss_cc + 
                      self.sim_weight * loss_sim + 
                      self.bce_weight * loss_bce +
                      self.nss_weight * loss_nss)

        return total_loss