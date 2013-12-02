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
    ps = []
    np = 10
    for i in range(np):
        p = Process(target=afdq, args = (working_dir, orig_dir, block_sizes, (num_data/np) + 1, parsed))
        p.start()
        ps.append(p)
    dsize = afdq(working_dir, orig_dir, block_sizes, (num_data/np) + 1, parsed)
    for p in ps:
        p.join()
    return dsize

def afdq(working_dir, orig_dir, block_sizes, num_data, parsed):
    dsize, file_list = addFakeDataQueue(working_dir, orig_dir, block_sizes, num_data)
    load_data_s3( file_list, working_dir, parsed['source-s3'])
    load_data_sqs( file_list,parsed['source-sqs'] )
    return dsize

def test_accuracy(orig_dir,  sqs_queue, s3_bucket, num_tests):
    s3 = boto.connect_s3()
    sqs = boto.connect_sqs()
    bucket = s3.get_bucket(s3_bucket)
    my_queue = sqs.get_queue(sqs_queue)
    while num_tests > 0:
        num_tests -= 1
        for m in my_queue.get_messages():
            res = json.loads(m.get_body())
            my_queue.delete_message(m)
        k = bucket.get_key(res['f_name'] ) 
        result_file = os.path.join( orig_dir, res['f_name'])
        k.get_contents_to_filename( result_file )
        rms = np.load( result_file )
        test = compare_serial_rms( orig_dir, res['file_id'], rms)
        print "Test", res['file_id'], "passed" if test else "failed"

def compare_serial_rms(orig_dir, file_id, rms_buffer)
    exp = np.load(os.path.join(orig_dir, '_'.join(['em',file_id,'original'])))
    nm = np.load(os.path.join(orig_dir, '_'.join(['nm',file_id,'original'])))
    sm = np.load(os.path.join(orig_dir, '_'.join(['sm',file_id,'original'])))
    gm = np.load(os.path.join(orig_dir, '_'.join(['gm',file_id,'original'])))

    _,_, rms = testDirac(em, gm, sm, nm)
    return np.allclose(rms, rms_buffer[:rms.shape[0],:rms.shape[1]])

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
        for t in ['add']:#'remove'
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
    parsed['heartbeat-interval'] = 10
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
    for k,v in dsize.iteritems():
        dsize[k] =v*10
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
            try:
                q.delete()    
            except:
                print "%s: won't delete" % q.name
def s3_cleanup(bucket= 'tcdirac-togpu-00'):
    conn = boto.connect_s3()
    b = conn.get_bucket(bucket)
    b.delete_keys(b.get_all_keys())

def testDirac(expression_matrix, gene_map, sample_map, network_map):
    srt = np.zeros((gene_map.shape[0]/2, expression_matrix.shape[1]))
    for i in range(expression_matrix.shape[1]):
        for j in range(gene_map.shape[0]/2):
            g1 = gene_map[2*j]
            g2 = gene_map[2*j +  1]
            if expression_matrix[g1,i] < expression_matrix[g2,i]:
                srt[j,i] = 1
            else:
                srt[j,i] = 0
    rt = np.zeros_like(srt)

    for i in range(expression_matrix.shape[1]):
        neigh = sample_map[i,:]
        t = srt[:,neigh].sum(axis=1)
        for j in range(len(t)):
            rt[j,i] = int(len(neigh)/2 < t[j])

    rms_matrix =  np.zeros_like(srt)
    for i in range(expression_matrix.shape[1]):
        for j in range(gene_map.shape[0]/2):
            rms_matrix[j,i] = int(rt[j,i] == srt[j,i])
    rms_final = np.zeros((len(network_map) - 1 , expression_matrix.shape[1]))

    for i in range(len(network_map) - 1):
        nstart = network_map[i]
        nend = network_map[i+1]
        rms_final[i,:] = rms_matrix[nstart:nend, :].sum(axis=0)/float(nend-nstart)
    return srt, rt, rms_final

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run_level', help="0:run, 1:load balance, 2:terminate, 3:cleanup")
    arg = parser.parse_args()
    num_data = 200 
    (RUN, LOAD_BALANCE, TERMINATE, CLEANUP) = range(4)
    runTest(num_data, level=arg.run_level)
