import data
import pycuda.driver as cuda
import numpy as np
from pycuda._driver import MemoryError
import logging 

def run( exp, gm, sm, nm, sample_block_size, npairs_block_size, nets_block_size, rms_only=True):
    srt = data.SharedSampleRankTemplate( exp.buffer_nsamples, gm.buffer_npairs )
    rt = data.SharedRankTemplate( exp.buffer_nsamples, gm.buffer_npairs )
    rms = data.SharedRankMatchingScores( nm.buffer_nnets, exp.buffer_nsamples )
    try:
        exp.toGPU( sample_block_size )
        rms.toGPU( sample_block_size, nets_block_size )
        nm.toGPU( nets_block_size )
        rt.toGPU( sample_block_size, npairs_block_size )
        sm.toGPU( sample_block_size )
        srt.toGPU( sample_block_size, npairs_block_size )
        gm.toGPU( npairs_block_size )
    except MemoryError:
        #we ran out of memory, maybe dev memory changed, in any case, 
        logging.error("*************MemoryERROR*********************")
        req_mem = reqMemory(exp, rms,np,rt,sm,srt,gm,nm, sample_block_size, nets_block_size, npairs_block_size )
        logging.error("Shared Dirac")
        logging.error( "Req. Mem[%f], Avail. Mem[%f]" % (float(req_mem)/1073741824.0, float(cuda.mem_get_info()[0])/1073741824.0) )
        for d in [exp,rms, nm, rt, sm, srt, gm]:
            if d.gpu_data is not None:
                d.gpu_data.free()
        raise
    dirac.sampleRankTemplate( exp.gpu_data, gm.gpu_data, srt.gpu_data, exp.buffer_nsamples, gm.buffer_npairs, npairs_block_size, sample_block_size)
    dirac.rankTemplate( srt.gpu_data, sm.gpu_data, rt.gpu_data, srt.buffer_nsamples, sm.buffer_kneighbors, gm.buffer_npairs, npairs_block_size, sample_block_size)
    dirac.rankMatchingScores( srt.gpu_data, rt.gpu_data, rms.gpu_data, nm.gpu_data, srt.buffer_nsamples, nm.buffer_nnets, sample_block_size, nets_block_size)
    return (srt, rt, rms)

def reqMemory(exp, rms,np,rt,sm,srt,gm,nm,sample_block_size, nets_block_size, npairs_block_size ):
    pred = exp.gpu_mem( sample_block_size )
    pred += rms.gpu_mem( sample_block_size, nets_block_size )
    pred += nm.gpu_mem( nets_block_size )
    pred += rt.gpu_mem( sample_block_size, npairs_block_size )
    pred += sm.gpu_mem( sample_block_size )
    pred += srt.gpu_mem( sample_block_size, npairs_block_size )
    pred += gm.gpu_mem( npairs_block_size )
    return pred

def sampleRankTemplate( exp_gpu, gmap_gpu, srt_gpu, nsamp, npairs, pairs_block_size, sample_block_size):
    """
    nsamp  is the columns dim (shape[1]) of exp_gpu
    npairs is the length of gmap_gpu (shape[0])
    """
    block = (pairs_block_size, sample_block_size, 1)
    grid = (npairs/pairs_block_size, nsamp/sample_block_size)
    kernel_source = kernels.srt(nsamp)
    mod = SourceModule(kernel_source)
    func = mod.get_function('srtKernel')
    func(exp_gpu, gmap_gpu, srt_gpu, block=block, grid=grid )

def rankTemplate(  srt_gpu, sample_map_gpu,rt_gpu, nsamples, neighbors, npairs, pairs_block_size, sample_block_size):
    """
    srt_gpu is (npairs, nsamples)
    sample_map_gpu is (neighbors, nsamples)
    """
    block = (pairs_block_size, sample_block_size, 1)
    grid = (npairs/pairs_block_size, nsamples/sample_block_size)
    kernel_source = kernels.rt( neighbors, nsamples )
    mod = SourceModule(kernel_source)
    func = mod.get_function('rtKernel')
    func( srt_gpu, sample_map_gpu, rt_gpu, block=block, grid=grid)

def rankMatchingScores( srt_gpu, rt_gpu, rms_gpu, nmap_gpu, nsamples, nnets, sample_block_size, nets_block_size):
    block = (nets_block_size, sample_block_size, 1)
    grid = ( nnets/nets_block_size, nsamples/sample_block_size)
    kernel_source = kernels.rms( nsamples, nnets )
    mod = SourceModule( kernel_source )
    func = mod.get_function('rmsKernel')
    func( rt_gpu, srt_gpu, nmap_gpu, rms_gpu, block=block, grid=grid )
