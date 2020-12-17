# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:10:57 2020

@author: Jisi
"""


from itertools import permutations

import torch
import torch.nn.functional as F

EPS = 1e-8


def cal_loss(source, estimate_source, source_lengths, cross_valid=False):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, 2**M, C, T]
        source_lengths: [B]
    """
    if not cross_valid:
        min_mse,  min_mse_idx = cal_logmse_with_mixit(source,
                                                  estimate_source,
                                                  source_lengths)
        loss = torch.mean(min_mse)
    else:
        max_snr,  max_snr_idx = cal_si_snr_with_mixit(source,
                                                  estimate_source,
                                                  source_lengths)  
        loss = 0 - torch.mean(max_snr)
    return loss, estimate_source
    #reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)
    #return loss, max_snr, estimate_source, reorder_estimate_source
    
def cal_logmse_with_mixit(source, estimate_source, source_lengths, max_snr=30):
    """Negative log MSE loss, the negated log of SNR denominator.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, 2**M, C, T]
        source_lengths: [B], each item is between [0, T]
        max_snr: SNR threshold
    """
    B, C, T = source.size()
    mask = get_mask(source, source_lengths)
    mask = torch.unsqueeze(mask, dim=1)
    estimate_source *= mask
    source = torch.unsqueeze(source, dim=1) # [B, 1, C, T]
    
    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1, 1).float()  # [B, 1, 1, 1]
    mean_target = torch.sum(source, dim=3, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=3, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask # [B, 1, C, T]
    zero_mean_estimate *= mask # [B, 2**M, C, T]
    
    snrfactor = 10.**(-max_snr / 10.)
    e_noise = zero_mean_estimate - zero_mean_target # [B, 2**M, C, T]
    target_pow = torch.sum(zero_mean_target ** 2, dim=3) + EPS # [B, 1, C]
    bias = snrfactor * target_pow
    snr_set = bias + torch.sum(e_noise ** 2, dim=3) # [B, 2**M, C]
    snr_set = 10 * torch.log10(snr_set + EPS)
    snr_set = torch.mean(snr_set, dim = 2)
    min_mse_idx = torch.argmin(snr_set, dim=1)  # [B]
    min_mse, _ = torch.min(snr_set, dim=1, keepdim=True)
    return min_mse, min_mse_idx

def cal_si_snr_with_mixit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, 2**M, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    #assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    mask = torch.unsqueeze(mask, dim=1)
    estimate_source *= mask
    source = torch.unsqueeze(source, dim=1) # [B, 1, C, T]

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1, 1).float()  # [B, 1, 1, 1]
    mean_target = torch.sum(source, dim=3, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=3, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask # [B, 1, C, T]
    zero_mean_estimate *= mask # [B, 2**M, C, T]

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = zero_mean_target  # [B, 1, C, T]
    s_estimate = zero_mean_estimate  # [B, 2**M, C, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, 2**M, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, 2**M, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, 2**M, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    # pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    # SNR threshold SNR(max) set as 30 dB
    # SI-SNR = 10 * log_10(||s_target||^2 / (||e_noise||^2 + tau*||s_target||^2))
    snr_clamp = 30
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + 
                        10 ** (- snr_clamp/10) * torch.sum(pair_wise_proj ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, 2**M, C]
    
    snr_set = torch.mean(pair_wise_si_snr, dim = 2)
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    #max_snr /= C
    return max_snr, max_snr_idx


def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask


if __name__ == "__main__":
    torch.manual_seed(123)
    B, C, T = 2, 3, 12
    # fake data
    source = torch.randint(4, (B, C, T))
    estimate_source = torch.randint(4, (B, C, T))
    source[1, :, -3:] = 0
    estimate_source[1, :, -3:] = 0
    source_lengths = torch.LongTensor([T, T-3])
    print('source', source)
    print('estimate_source', estimate_source)
    print('source_lengths', source_lengths)
    
    loss, max_snr, estimate_source, reorder_estimate_source = cal_loss(source, estimate_source, source_lengths)
    print('loss', loss)
    print('max_snr', max_snr)
    print('reorder_estimate_source', reorder_estimate_source)
