# Pytorch-Face-recongition-state-of-the-art-Qmul-surveface-

Implementation state of the art for face recognition algorithm.


In my repos, I use timm module to quickly create model ( efficienet, resnet ...) and reference some notebook from kaggle.
Implementation some SOTA algorithm for recognition: Arcface, adaptive arcface, adacos, additive margin,... 
Optimizer: SAM optimizer


We only config on class CFG in file train.py:


      path_train: train struture = train/ class1
                                        / class2
                                        ....


      path_valid: train struture = valid/ class1
                                        / class2
                                        ....
      and some config for optimizer , loss, ...
