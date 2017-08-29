# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 18:53:09 2017

@author: Jannik
"""
# Test Texture Arrays
#texture = [66,58,56,64,62,68,66,63]
#texture = [182,5,12,250,193,80,248,126]
texture = [248,12,5,250,126,80,182,193]

P = 8 # Pixelcount of the Kernel

### Implementation of the VAR Lbp in Skimage ###
lbp = 0
sum_ = 0.0
var_ = 0.0
for i in range(P):
    texture_i = texture[i] #gp
    sum_ += texture_i
    var_ += texture_i * texture_i
var_ = (var_ - (sum_ * sum_) / P) / P
if var_ != 0:
    lbp = var_
else:
    lbp = 0
    
print(lbp)


### Implementation of the VAR Lbp by Jannik ###
var_ = 0.0
sum_ = 0.0
for i in range(P):
    u = 0.0
    for j in range(P):
        u += texture[j]
    u = u / P
    
    sum_ += (texture[i] - u)**2

var_ = sum_ / P
    
print(var_)


### Implementation of the NI-LBP by Jannik ###
niLBP = []
sum_ = 0.0
for i in range(P):
    u = 0.0
    for j in range(P):
        u += texture[j]
    u = u / P
    
    sum_ = (texture[i] - u)
    if sum_ >= 0:
        sum_ = 1
    else:
        sum_ = 0
    
    sum_ = sum_ * 2**i
    
    niLBP.append(sum_)

print(sum(niLBP))    
print(niLBP)    
    