from multiprocessing import Process, Event
from Queue import Empty, Full

import boto
from boto.s3.key import Key
import boto.sqs
from boto.sqs.message import Message

import os
import os.path

from gpudirac.utils import static
import logging
import time
import json
import random

class Poster(Process):
    """
    Takes the names of files that have been written by a Packer from an mp queue
    and copies that file to s3.
    It then notifies the rest of the cluster via sqs that this has occured.
    """
    def __init__(self, name, out_dir, in_dir, q_gpu2s3, evt_death, sqs_name, in_sqs_name, s3bucket_name):
        Process.__init__(self, name=name)
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.logger.debug( "Init: out_dir<%s> sqs_name<%s> s3bucket_name<%s>" % (out_dir, sqs_name, s3bucket_name) )
        self.q_gpu2s3 = q_gpu2s3
        self.sqs_name = sqs_name
        self.in_dir = in_dir
        self._sqs_q = self._connect_sqs(sqs_name)
        #note clean up in_sqs mess
        self._sqs_conn = boto.sqs.connect_to_region('us-east-1') 
        self._in_sqs_q = self._sqs_conn.get_queue(in_sqs_name)
        self.s3bucket_name = s3bucket_name
        self._s3_bucket = self._connect_s3()
        self.out_dir = out_dir
        self.evt_death = evt_death

    def run(self):
        self.logger.info("starting...")
        while not self.evt_death.is_set():
            self.run_once()

    def run_once(self):
        try:
            f_info = self.q_gpu2s3.get(True, 3)
            self.upload_file( f_info['f_name'] )
            m = Message(body= json.dumps(f_info) )
            self._sqs_q.write( m )
            self._delete_message( f_info['file_id'] )
        except Empty:
            #self.logger.info("starving")
            if self.evt_death.is_set():
                self.logger.info("Exiting...")
                return
        except:
            self.logger.exception("exception in run_once")
            self.evt_death.set()

    def _delete_message(self, file_id):
        with open(os.path.join(self.in_dir, 'receipt_handle_' + file_id), 'r') as fh:
            file_handle = fh.read()
        self._sqs_conn.delete_message_from_handle(self._in_sqs_q, file_handle)

    def _connect_s3(self):
        conn = boto.connect_s3()        
        b = conn.get_bucket( self.s3bucket_name )
        return b

    def _connect_sqs(self, name=None):
        if name is None:
            name = self.sqs_name
        conn = boto.sqs.connect_to_region('us-east-1')
        q = conn.create_queue( name )
        return q

    def upload_file(self, file_name):
        k = Key(self._s3_bucket)
        k.key = file_name
        k.set_contents_from_filename( os.path.join(self.out_dir, file_name), reduced_redundancy=True )
        self.logger.debug("Deleting <%s>" % (os.path.join(self.out_dir, file_name)))
        os.remove(os.path.join(self.out_dir, file_name))

class PosterQueue:
    def __init__(self,  name, out_dir,  q_gpu2s3,sqs_name, s3bucket_name, in_dir, in_sqs):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.logger.debug( "Init: out_dir<%s> sqs_name<%s> s3bucket_name<%s>"% (out_dir, sqs_name, s3bucket_name) )
        self.out_dir = out_dir
        self.q_gpu2s3= q_gpu2s3
        self.sqs_name = sqs_name
        self.s3bucket_name = s3bucket_name
        self._posters = []
        self._reaper = []
        self.in_dir = in_dir
        self.in_sqs_q = in_sqs

    def add_poster(self, num=1):
        if num <= 0:
            return
        else:
            evt_death = Event()
            evt_death.clear()
            self._posters.append( Poster(self.name + "_Poster_" + str(num), self.out_dir, self.in_dir,  self.q_gpu2s3, evt_death, self.sqs_name,self.in_sqs_q, self.s3bucket_name))
            self._reaper.append(evt_death)
            self._posters[-1].daemon = True
            self._posters[-1].start()
            self.add_poster(num - 1)

    def remove_poster(self):
        if len(self._posters) == 0:
            raise Exception("Attempt to remove poster from empty queue")
        self.logger.info("removing poster")
        self._reaper[-1].set()
        ctr = 0
        while self._posters[-1].is_alive() and ctr < 20:
            time.sleep(.2)
            ctr += 1
        if self._posters[-1].is_alive():
            self._posters[-1].terminate()
        self._reaper = self._reaper[:-1]
        self._posters = self._posters[:-1]

    def repair(self):
        for i, d in enumerate(self._reaper):
            if d.is_set():
                p = self._posters[i]
                if p.is_alive():
                    p.terminate()
                p.join(.5)    
            d.clear()
            self.logger.warning("Repairing poster<%i>" % i)
            self._posters[i] =  Poster(self.name + "_" + str(i)+"_repaired", self.out_dir,  self.q_gpu2s3, d, self.sqs_name, self.s3bucket_name)

    def _sp_alive(self):
        for d in self._reaper:
            if d.is_set():
                return False
        return False

    def kill_all(self):
        self.logger.info("sending death signals")
        for r in self._reaper:
            r.set()

    def clean_up(self):
        for r in self._posters:
            if r.is_alive():
                time.sleep(2)
                r.terminate()
        for r in self._posters:
            r.join()
        self._posters = []
        self._reaper = []
        self.logger.info("Complete...")

    def num_sub(self):
        count = 0
        for r in self._posters:
            if r.is_alive():
                count += 1
        return count
