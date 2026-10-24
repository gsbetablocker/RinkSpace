"""GPU implementation of the same continuous-board station/height operator.

The integral nodes, weights, segment breaks and arclengths come directly from
ContinuousBoard. There is no alternative approximate table or physical model.
"""
import ctypes as C
from pathlib import Path
import numpy as np
from .compute import CudaDevice

def pack_board(board):
    if len(board.nodes)!=128 or len(board.weights)!=128:
        raise ValueError('CUDA board integral requires the current explicit 128-node definition')
    result=np.r_[board.width,board.rx,board.ry,board.exponent,board.height,
                 board.corner_length,board.lengths,board.breaks].astype(np.float64)
    if result.shape!=(23,) or not np.isfinite(result).all():raise ValueError('current board ABI shape/values')
    return result

class CudaBoardBatch:
    def __init__(self,count,*,variants=1,device=None,generation=1):
        if type(count) is not int or count<=0 or type(variants) is not int or variants<=0:raise ValueError('positive integer batch sizes required')
        self.device=device or CudaDevice();self.owns_device=device is None
        self.count=count;self.variants=variants;self.generation=generation
        self.q=self.device.allocate((variants,count,2),np.float64,generation=generation)
        self.boards=self.device.allocate((variants,23),np.float64,generation=generation,role='parameter')
        self.nodes=self.device.allocate((128,),np.float64,generation=generation)
        self.weights=self.device.allocate((128,),np.float64,generation=generation)
        self.output=self.device.allocate((variants,count,6),np.float64,generation=generation)
        self.status=self.device.allocate((variants,count),np.uint8,generation=generation)
        try:
            kernels,self.compilation=self.device.compile(Path(__file__).with_name('board_at.cu'),('board_at',))
        except Exception:self.close();raise
        self.kernel=kernels['board_at'];self.kernel_milliseconds=None
    def evaluate(self,boards,station_height,*,measure=False):
        if self.variants==1 and not isinstance(boards,(list,tuple)):boards=[boards]
        if len(boards)!=self.variants:raise ValueError('board variant count changed; allocate the current batch layout explicitly')
        q=np.asarray(station_height)
        if q.dtype!=np.dtype('float64'):raise ValueError('station/height ABI must be explicit float64')
        if q.shape==(self.count,2):q=np.broadcast_to(q,(self.variants,self.count,2))
        if q.shape!=(self.variants,self.count,2) or not np.isfinite(q).all():raise ValueError('finite native station/height shape required')
        nodes,weights=np.asarray(boards[0].nodes),np.asarray(boards[0].weights)
        if any(not np.array_equal(b.nodes,nodes) or not np.array_equal(b.weights,weights) for b in boards):raise ValueError('mixed quadrature definitions in one board batch')
        d=self.device;g=self.generation
        d.update(self.q,np.ascontiguousarray(q),generation=g)
        d.update(self.boards,np.stack([pack_board(b) for b in boards]),generation=g)
        d.update(self.nodes,np.ascontiguousarray(nodes),generation=g);d.update(self.weights,np.ascontiguousarray(weights),generation=g)
        def launch():d.launch(self.kernel,self.count*self.variants,(self.q,self.boards,self.nodes,self.weights,C.c_int(self.count),C.c_int(self.variants),self.output,self.status))
        if measure:self.kernel_milliseconds=d.timed(launch)
        else:launch()
        value=d.download(self.output);status=d.download(self.status)
        return {'xyz':value[...,:3],'jacobian_station':value[...,3:6],'status':status,'valid':status==0}
    def close(self):
        for name in ('q','boards','nodes','weights','output','status'):
            buffer=getattr(self,name,None)
            if buffer is not None:buffer.close()
        if self.owns_device:self.device.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
