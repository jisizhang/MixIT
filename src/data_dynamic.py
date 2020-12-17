# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 19:42:18 2020

Dataset for source separation and speech enhancement with dynamic mixing

@author: Jisi
"""

import json
import math
import os
import random

import numpy as np
import torch
import torch.utils.data as data

import soundfile as sf

class DynamicAudioDataset(data.Dataset):

    def __init__(self, json_dir,
                 sample_rate=8000, segment=4.0,
                 cv_maxlen=8.0, dynamic=True):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(DynamicAudioDataset, self).__init__()
        self.dynamic = dynamic
        self.segment_len = int(segment * sample_rate)
        mix_json = os.path.join(json_dir, 'mix.json')
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            s2_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        sorted_s1_infos = sort(s1_infos)
        sorted_s2_infos = sort(s2_infos)
        if segment >= 0.0:
            # segment length and count dropped utts
            drop_utt, drop_len = 0, 0
            for i in range(len(sorted_mix_infos)- 1, -1, -1): # Go backward
                sample = sorted_mix_infos[i][1]
                if sample < self.segment_len:
                    drop_utt += 1
                    drop_len += sample
                    del sorted_mix_infos[i]
                    del sorted_s1_infos[i]
                    del sorted_s2_infos[i]
            print("Drop {} utts({:.2f} h) which is short than {} samples".format(
                drop_utt, drop_len/sample_rate/36000, self.segment_len))
        self.mix_infos = sorted_mix_infos
        self.s1_infos = sorted_s1_infos
        self.s2_infos = sorted_s2_infos

    def __getitem__(self, index):
        if not self.dynamic:
            mix_file, mix_len = self.mix_infos[index]
            s1_file, s1_len = self.s1_infos[index]
            s2_file, s2_len = self.s2_infos[index]
            if self.segment_len < mix_len:
                offset = np.random.randint(0, mix_len - self.segment_len)
                mix, _ = sf.read(mix_file, start=offset, stop=offset + self.segment_len, dtype="float32") # frames x channels
                s1, _ = sf.read(s1_file, start=offset, stop=offset + self.segment_len, dtype="float32")
                s2, _ = sf.read(s2_file, start=offset, stop=offset + self.segment_len, dtype="float32")
            mix = mix.T
            s1 = s1.T
            s2 = s2.T
            sources = torch.from_numpy(np.vstack((s1[0,:], s2[0,:])))
            return torch.from_numpy(mix[0,:]), torch.as_tensor(self.segment_len), sources
        
        mixtures = random.sample(self.mix_infos, k=2)
        sources = []
        for mix_path, mix_len in mixtures:
            offset = 0
            if self.segment_len < mix_len:
                offset = np.random.randint(0, mix_len - self.segment_len)
            tmp, _ = sf.read(mix_path, start=offset, stop=offset + self.segment_len, dtype="float32") # frames x channels
            tmp = tmp.T
            sources.append(tmp[0,:])
        mix = np.sum(np.stack(sources), 0)
        
        # check for clipping
        absmax = np.max(np.abs(mix))
        if absmax > 1:
            mix = mix / absmax
            sources = [x / absmax for x in sources]
            
        sources = np.stack(sources)
        return torch.from_numpy(mix).float(), torch.as_tensor(self.segment_len), torch.from_numpy(sources).float()

    def __len__(self):
        return len(self.mix_infos)