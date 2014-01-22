from multiprocessing import Process,  Event
from Queue import Empty, Full

import boto
from boto.s3.key import Key
import boto.sqs
from boto.sqs.message import Message

import os
import os.path

import gpudirac
from gpudirac.utils import static
import logging
import time
import json
import random

class Retriever(Process):
    """
    Monitors sqs for new input data.
    When data is available, downloads the data and alerts
    """
    def __init__(self, name, in_dir,  q_ret2gpu, evt_death, sqs_name, s3bucket_name, max_q_size):
        Process.__init__(self, name=name)
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.logger.debug( "Init: in_dir<%s> sqs_name<%s> s3bucket_name<%s> max_q_size<%i>"% (in_dir, sqs_name, s3bucket_name, max_q_size) )
        self.q_ret2gpu = q_ret2gpu
        self.sqs_name = sqs_name
        self._sqs_q = self._connect_sqs()
        self.s3bucket_name = s3bucket_name
        self._s3_bucket = self._connect_s3()
        self.in_dir = in_dir
        self.evt_death = evt_death
        self.max_q_size = max_q_size
        self.bad_file_cache = {}
        self.num_messages_pull = 1
        self.visibility_timeout = 600

    def run(self):
        while not self.evt_death.is_set():
            if self.q_ret2gpu.qsize() < self.max_q_size:
                messages = self.run_once()
            if messages < self.num_messages_pull and not self.evt_death.is_set():
                self.logger.warning("starving")
                time.sleep(random.randint(1,10))

    def run_once(self):
        """
        Does the work
        """
        messages = self._sqs_q.get_messages(self.num_messages_pull, 
                    visibility_timeout=self.visibility_timeout)
        m_count = 0
        for message in messages:
            try:
                self.logger.debug("Recvd: %r" % message)
                self.logger.debug("Body: %s" % message.get_body())
                m = json.loads(message.get_body())
                error = False 
                for f_type, f in m['f_names'].iteritems():
                    if self.download_file( f ):
                        self.logger.debug("Downloaded <%s>" % f)
                        m[f_type] = f
                        self._write_receipt_handle( m['file_id'], message.receipt_handle )
                    else:
                        self.logger.warning("FileMissing: <%s> does not exist on s3" % f)
                        if f in self.bad_file_cache and Message is not None:
                            #seen this bad file before, fool me once ...
                            self._sqs_q.delete_message(message)
                            #so another file in the package does not try to  delete the message
                        else:
                            self.bad_file_cache[f] = 1
                            message = None
                cont = True
                while cont and message is not None:
                    try:
                        self.q_ret2gpu.put( m, timeout=10 )
                        cont = False
                    except Full:
                        self.logger.warning("queue_full" )
                        if self.evt_death.is_set():
                            return m_count
                m_count += 1
            except:
                self.logger.exception("While trying to download files" )
        return m_count
    def _write_receipt_handle(self, file_id, handle): 
        f_out = os.path.join(self.in_dir, 'receipt_handle_' + file_id)
        with open(f_out,'w') as rhf:
            rhf.write(handle)

    def _connect_s3(self):
        conn = boto.connect_s3()
        b = conn.get_bucket( self.s3bucket_name )
        return b

    def _connect_sqs(self):
        conn = boto.sqs.connect_to_region('us-east-1')
        q = conn.get_queue( self.sqs_name )
        return q

    def download_file(self, file_name):
        k = Key(self._s3_bucket)
        k.key = file_name
        local_path =  os.path.join(self.in_dir, file_name)
        if os.path.exists(local_path):
            return True
        elif k.exists():
            k.get_contents_to_filename( local_path )
            return True
        else:
            return False

class RetrieverQueue:
    """
    Manages a set of retrievers
    """
    def __init__(self,  name, in_dir,  q_ret2gpu,sqs_name, s3bucket_name):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.logger.debug( "Init: in_dir<%s> sqs_name<%s> s3bucket_name<%s>"% (in_dir, sqs_name, s3bucket_name) )
        self.in_dir = in_dir
        self.q_ret2gpu= q_ret2gpu
        self.sqs_name = sqs_name
        self.s3bucket_name = s3bucket_name
        self._retrievers = []
        self._reaper = []

    def add_retriever(self, num=1):
        if num <= 0:
            return
        else:
            evt_death = Event()
            evt_death.clear()
            self._retrievers.append( Retriever(self.name + "_Retriever_" + str(num), self.in_dir,  self.q_ret2gpu, evt_death, self.sqs_name, self.s3bucket_name, max_q_size=10*(num+1)  ) )
            self._reaper.append(evt_death)
            self._retrievers[-1].daemon = True
            self._retrievers[-1].start()
            self.add_retriever(num - 1)

    def remove_retriever(self):
        if len(self._retrievers) == 0:
            raise Exception("Attempt to remove retriever from empty queue")
        self.logger.info("removing retriever")
        self._reaper[-1].set()
        ctr = 0
        while self._retrievers[-1].is_alive() and ctr < 10:
            time.sleep(.2)
            ctr += 1
        if self._retrievers[-1].is_alive():
            self._retrievers[-1].terminate()
        self._reaper = self._reaper[:-1]
        self._retrievers = self._retrievers[:-1]

    def repair(self):
        for i, d in enumerate(self._reaper):
            if d.is_set():
                p = self._retrievers[i]
                if p.is_alive():
                    p.terminate()
                p.join(.5)    
            d.clear()
            self._retrievers[i] =  Retriever(self.name + "_Retriever_" + str(i)+"_repaired", self.in_dir,  self.q_ret2gpu, d, self.sqs_name, self.s3bucket_name, max_q_size=10*i)

    def kill_all(self):
        for r in self._reaper:
            r.set()

    def all_dead(self):
        for r in self._retrievers:
            if r.is_alive():
                return False
        return True

    def clean_up(self):
        for r in self._retrievers:
            if r.is_alive():
                r.terminate()
        for r in self._retrievers:
            r.join()
        self._retrievers = []
        self._reaper = []

    def num_sub(self):
        count = 0
        for r in self._retrievers:
            if r.is_alive():
                count += 1
        return count

