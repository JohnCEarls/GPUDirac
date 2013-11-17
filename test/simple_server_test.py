import boto
import boto.sqs
from boto.sqs.message import Message
import json
import time

def mockMaster( master_q = 'tcdirac-master'):
    try:
        conn = boto.sqs.connect_to_region( 'us-east-1' )
        in_q = conn.get_queue( master_q )
        m = None
        while m is None:
            print "MM: waiting for message.. "
            m = in_q.read( wait_time_seconds=20 )
        in_q.delete_message(m)
        settings = json.loads(m.get_body())
        print "MM: ", str(settings)
        rq = conn.get_queue( settings['response'] )
        cq = conn.get_queue( settings['command'] )

        m = Message(body=get_gpu_message())

        cq.write(m)

        time.sleep(10)

        for m in get_lb_messages():
            cq.write(Message(body=json.dumps(m)))
        print "MM: Sending terminate signal"
        cq.write( Message(body=get_terminate_message()))
    except:
        print "*"*30
        print "Error in mockMaster"
        print "*"*30
        raise
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

def get_gpu_message():
    parsed = {}
    parsed['result-sqs'] = 'tcdirac-from-gpu-00'
    parsed['source-sqs'] = 'tcdirac-to-gpu-00'
    parsed['source-s3'] = 'tcdirac-togpu-00'
    parsed['result-s3'] = 'tcdirac-fromgpu-00'
    dsize = {'em':10000, 'gm':1000, 'sm':1000, 'nm':1000, 'rms':1000}
