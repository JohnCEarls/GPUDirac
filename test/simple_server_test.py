import boto
import boto.sqs
from boto.sqs.message import Message
import json
import time
import os, os.path

import sys

import errno
import time
import itertools
from tempfile import TemporaryFile
import hashlib
import random
import cPickle
from gpudirac.utils import dtypes
from gpudirac.device import data

from multiprocessing import Process, Queue, Lock, Value, Event, Array
from Queue import Empty

import boto
from boto.s3.key import Key
from boto.sqs.connection import SQSConnection
from boto.sqs.message import Message

import ctypes
import numpy as np
import scipy.misc
import pandas

import pycuda.driver as cuda


def runTest(num_data, level=0 ):
    if level == 0:
        #creates data and starts server
        dsize = push_data( num_data)
        init_signal(dsize)
    elif level == 1:
        #load balance
        settings = None
        with open('settings.json', 'r') as s:
            settings = json.loads(s.read())
        if settings is None:
            raise Exception("Shit on Load Balance")
        load_balance_signal( settings['command'] )
    elif level == 2:
        with open('settings.json', 'r') as s:
            settings = json.loads(s.read())
        if settings is None:
            raise Exception("Shit on Load Balance")
        terminate_signal( settings['command'] ) 
    elif level == 3:
        sqs_cleanup()
        s3_cleanup(bucket= 'tcdirac-togpu-00')    

def push_data(num_data):

    block_sizes = (32,16,8)
    working_dir = '/scratch/sgeadmin/working'
    orig_dir = '/scratch/sgeadmin/original'
    parsed = {}
    parsed['result-sqs'] = 'tcdirac-from-gpu-00'
    parsed['source-sqs'] = 'tcdirac-to-gpu-00'
    parsed['source-s3'] = 'tcdirac-togpu-00'
    parsed['result-s3'] = 'tcdirac-fromgpu-00'

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    if not os.path.exists(orig_dir):
        os.makedirs(orig_dir)
    dsize, file_list = addFakeDataQueue(working_dir, orig_dir, block_sizes, num_data)
    load_data_s3( file_list, working_dir, parsed['source-s3'])
    load_data_sqs( file_list,parsed['source-sqs'] )
    return dsize

def init_signal(dsize,  master_q = 'tcdirac-master'):
    try:
        conn = boto.sqs.connect_to_region( 'us-east-1' )
        in_q = conn.get_queue( master_q )
        m = None
        while m is None:
            print "MM: waiting for message.. "
            m = in_q.read( wait_time_seconds=20 )
        in_q.delete_message(m)
        settings = json.loads(m.get_body())
        cq = conn.get_queue( settings['command'] )

        m = Message(body=get_gpu_message(dsize))

        cq.write(m)

        time.sleep(10)
        print "settings"
        settings['master_q'] = master_q
        with open('settings.json', 'w') as s:
            s.write(json.dumps( settings ))
    except:
        print "*"*30
        print "Error in mockMaster"
        print "*"*30
        raise

def terminate_signal( command_q ): 
    conn = boto.sqs.connect_to_region( 'us-east-1' )
    cq = conn.get_queue( command_q )
    cq.write( Message(body=get_terminate_message()))
    
def load_balance_signal( command_q ):
    conn = boto.sqs.connect_to_region( 'us-east-1' )
    cq = conn.get_queue( command_q )
    for m in get_lb_messages():
        cq.write(Message(body=json.dumps(m)))

def get_lb_messages():
    mess = []
    for p in ['loader','poster','packer', 'retriever']:
        for t in ['add','remove']:
            command = {}
            command['message-type'] = 'load-balance'
            command['process'] = p 
            command['type'] = t
            command['increment'] = 3
            command['min'] = 2
            mess.append(command)
    return mess

def get_gpu_message(dsize):
    parsed = {}
    parsed['result-sqs'] = 'tcdirac-from-gpu-00'
    parsed['source-sqs'] = 'tcdirac-to-gpu-00'
    parsed['source-s3'] = 'tcdirac-togpu-00'
    parsed['result-s3'] = 'tcdirac-fromgpu-00'
    dtype = {'em':np.float32, 'gm':np.int32, 'sm':np.int32,'nm':np.int32,'rms':np.float32 }
    ds = []
    for k in ['em', 'gm', 'sm', 'nm']:
        ds.append( (k, dsize[k], dtypes.to_index(dtype[k])))

    parsed['data-settings'] = {'source':ds}
    ds = [('rms', dsize['rms'], dtypes.to_index(dtype['rms'])) ]
    parsed['data-settings']['results'] = ds
    parsed['gpu-id'] = 0

    parsed['sample-block-size'] = 32
    parsed['pairs-block-size'] = 16
    parsed['nets-block-size'] = 8

    parsed['heartbeat-interval'] = 1
    return json.dumps(parsed)

def get_terminate_message():
    parsed = {}
    parsed['message-type'] = 'termination-notice'
    return json.dumps(parsed)


def addFakeDataQueue(in_dir,orig_dir, block_sizes, num_data): 
    sample_block_size, npairs_block_size, nets_block_size = block_sizes
    unique_fid = set()
    buffer_nsamp = 0
    buffer_nnets = 0
    check_list = []
    dsize = {'em':10000, 'gm':1000, 'sm':1000, 'nm':1000, 'rms':1000} 
    for i in range(num_data):
        fake = genFakeData( 200, 20000)
        p_hash = None
        for i in range(1):
            f_dict = {}
            f_id = str(random.randint(10000,100000))
            while f_id in unique_fid:
                f_id = str(random.randint(10000,100000))
            f_dict['file_id'] = f_id
            unique_fid.add(f_id)

            for k,v in fake.iteritems():
                if k == 'em':
                    exp = data.Expression(v)
                    exp.createBuffer(sample_block_size, buff_dtype=np.float32)
                    v = exp.buffer_data
                    t_nsamp = exp.orig_nsamples
                    buffer_nsamp = exp.buffer_nsamples
                elif k == 'sm':
                    sm = data.SampleMap(v)
                    sm.createBuffer(sample_block_size, buff_dtype=np.int32)
                    v = sm.buffer_data
                elif k == 'gm':
                    gm = data.GeneMap(v) 
                    gm.createBuffer( npairs_block_size, buff_dtype=np.int32)
                    v = gm.buffer_data
                elif k == 'nm':
                    nm = data.NetworkMap(v)
                    nm.createBuffer( nets_block_size, buff_dtype=np.int32 )
                    v = nm.buffer_data
                    buffer_nnets = nm.buffer_nnets
                f_hash = hashlib.sha1(v).hexdigest()
                if k == 'em':
                    if p_hash is None:
                        p_hash = f_hash
                        p_temp = v.copy()
                    else:
                        assert p_hash == f_hash, str(v) + " " + str(p_temp)
                        p_hash = None
                        p_temp = None
                f_name = '_'.join([ k, f_dict['file_id'], f_hash])
                with open(os.path.join( in_dir, f_name),'wb') as f:
                    np.save(f, v)
                
                if v.size > dsize[k]:
                    dsize[k] = v.size
                f_dict[k] = f_name

            rms_buffer_data_size = int(buffer_nnets * buffer_nsamp * np.float32(1).nbytes * 3.1) #10% bigger than needed
            if rms_buffer_data_size > dsize['rms']:
               dsize['rms'] = rms_buffer_data_size 
            check_list.append(f_dict)
            for k,v in fake.iteritems():
                #save original data
                f_path = os.path.join(orig_dir, '_'.join([k,f_dict['file_id'],'original']))
                with open(f_path, 'wb') as f:
                    np.save( f, v )
    return dsize, check_list


def genFakeData( n, gn):
    neighbors = random.randint(5, 20)
    nnets = random.randint(50,100)

    samples = map(lambda x:'s%i'%x, range(n))
    genes = map(lambda x:'g%i'%x, range(gn))
    g_d = dict([(gene,i) for i,gene in enumerate(genes)])
    gm_text = []
    gm_idx = []


    exp = np.random.rand(len(genes),len(samples)).astype(np.float32)
    exp_df = pandas.DataFrame(exp,dtype=float, index=genes, columns=samples)

    net_map = [0]

    for i in range(nnets):
        n_size = random.randint(5,100)

        net_map.append(net_map[-1] + scipy.misc.comb(n_size,2, exact=1))
        net = random.sample(genes,n_size)
        for g1,g2 in itertools.combinations(net,2):
            gm_text.append("%s < %s" % (g1,g2))
            gm_idx += [g_d[g1],g_d[g2]]

    #data
    expression_matrix = exp
    gene_map = np.array(gm_idx).astype(np.uint32)
    #print gene_map[:20]
    sample_map = np.random.randint(low=0,high=len(samples), size=(len(samples),neighbors))
    sample_map = sample_map.astype(np.uint32)
    #print sample_map[:,:3]
    network_map = np.array(net_map).astype(np.uint32)

    data = {}
    data['em'] = expression_matrix
    data['gm'] = gene_map
    data['sm'] = sample_map
    data['nm'] = network_map
    return data


def load_data_s3(file_list, working_dir, in_s3_bucket):
    conn = boto.connect_s3()
    bucket = conn.create_bucket(in_s3_bucket)
    for f_dict in file_list:
        for k, f_name in f_dict.iteritems():
            if k in ['em','gm','sm','nm']:
                f_path = os.path.join( working_dir, f_name)
                k = Key(bucket)
                k.key = f_name
                k.set_contents_from_filename(f_path)

def load_data_sqs( file_list, in_sqs_queue):
    conn = boto.connect_sqs()
    q = conn.create_queue(in_sqs_queue)
    for f_dict in file_list:
        t = {}
        t['file_id'] = f_dict['file_id']
        t['f_names'] = []
        for k,v in f_dict.iteritems():
            if k in ['em','sm','nm','gm']:
                t['f_names'].append(v)        
        m = Message(body=json.dumps(t))
        q.write(m)

def sqs_cleanup():
    conn = boto.connect_sqs()
    constant = ['tcdirac-from-gpu-00','tcdirac-master', 'tcdirac-to-gpu-00']
    for q in conn.get_all_queues():
        if q.name in constant:
            q.clear()
        else:
            q.delete()    
def s3_cleanup(bucket= 'tcdirac-togpu-00'):
    conn = boto.connect_s3()
    b = conn.get_bucket(bucket)
    b.delete_keys(b.get_all_keys())


    
if __name__ == "__main__":
    num_data = 10
    (RUN, LOAD_BALANCE, TERMINATE, CLEANUP) = range(4)
    #s3_cleanup()
    runTest(num_data, level=TERMINATE)
