# -*- coding: utf-8 -*-
"""Functions for FIR filter design."""
import torch
import numpy as np

def firwin(numtaps, cutoff,  window='hamming', scale=True):
    '''
    only for low-pass filter
    :param numtaps: int, length of window
    :param cutoff:  cut-off frequency, between (0, 1), size = (n, 1)
    :param window:  type of window, only 'hamming' and 'hanning' are available here
    :param scale:   scale or not, bool
    :return:        kernel of low-pass filter
    '''
    fs = 2
    nyq = 0.5 * fs
    cutoff = cutoff / float(nyq) # (n, 1)
    # bands = torch.cat([torch.zeros_like(cutoff, device=cutoff.device), cutoff], dim=1)  # (n, 2)

    # Build up the coefficients.
    alpha = 0.5 * (numtaps - 1)
    m = torch.arange(0, numtaps, device=cutoff.device) - alpha
    m = m.reshape(1, -1) # (1, numtabs)

    # only computing the right band for low-pass filter (left is 0)
    h = cutoff * torch.sinc(cutoff * m) # (n, numtabs)

    # Get and apply the window function.
    if window == 'hamming':
        win = torch.tensor([0.54 - 0.46 * np.cos(2 * np.pi * n / (numtaps - 1)) for n in range(numtaps)], device=cutoff.device)
    elif window == 'hanning':
        win = torch.tensor([0.5 - 0.5 * np.cos(2 * np.pi * n / (numtaps - 1)) for n in range(numtaps)], device=cutoff.device)
    else:
        raise ValueError('type of window should be "hamming" or "hanning", got {}'.format(window))
    win = win.reshape(1, -1) # (1, numtabs)
    h = h * win

    # Now handle scaling if desired.
    if scale:
        s = torch.sum(h, dim=1, keepdim=True)  # (n, 1)
        h = h / s

    return h
