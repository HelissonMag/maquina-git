# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:25:41 2019

@author: homcerqueira
"""
import seaborn as sns

x = [[0,1,0,0,1,0],
     [0,1,0,0,1,0],
     [0,1,0,0,1,0],
     [0,1,1,1,1,0],
     [0,1,0,0,1,0],
     [0,1,0,0,1,0],
     [0,1,0,0,1,0],]
ax = sns.heatmap(x)
