# -*- coding: utf-8 -*-

class LFrf():
    def __init__(self,seq,min_steps:int=4):
        self.v1=seq.v1[:,0]
        self.phase=seq.phase[:,0]
        self.voff=seq.voff
        self.v0=seq.expsys.v0
        