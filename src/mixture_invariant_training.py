# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:55:53 2020

Mixture Invariant Training Criterion Implementation

@author: Jisi
"""

import numpy as np

"""
Exhaustive search for A:
    2xM binaray matrices, the set of matrices which assign each source s_m_hat
    to either x_1 and x_2
"""
M = 8

nums = np.arange(2**M)
bin_nums = ((nums.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
print("\nBinary representation of the said vector:")
print(bin_nums[:,::-1])
bin_nums = bin_nums[:,::-1]
vec_one = np.ones((M,), dtype=int)
A = np.zeros((2**M, 2, M), dtype=int)

for i in range(2**M):
    A[i,0,:] = bin_nums[i,:]
    A[i,1,:] = vec_one - bin_nums[i,:]