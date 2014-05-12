import time
import signal
import sys
import os, os.path
from multiprocessing import Queue,Process
import multiprocessing
import logging
import socket
import numpy as np
import boto
import boto.utils
from boto.s3.key import Key
import boto.sqs
from boto.sqs.connection import SQSConnection
from boto.sqs.message import Message
import random
import json
import pycuda.driver as cuda

from subprocesses.retriever import RetrieverQueue
from subprocesses.poster import PosterQueue
from subprocesses.loader import LoaderQueue, MaxDepth
from subprocesses.packer import PackerQueue
from device import dirac
from device import data
from utils import static, debug, dtypes

class PosterProgress(Exception):
    pass

class Dirac:
    """
    Class for running dirac on the gpu.
    name: a unique identifier for this dirac instance
    directories: contains local storage locations
        directories['source'] : data source directorie
        directories['results'] : directory for writing processed data
        directories['log'] : directory for logging
    #these settings are retrieved from cluster master via the command queue
    s3: dict containing names of buckets for retrieving and depositing data
        s3['source'] : bucket for source data
        s3['results'] : bucket for writing results
    sqs: dict contain sqs queues for communication with data nodes and commands
        sqs['source'] : queue containing data files ready for download
        sqs['results'] : queue for putting result data name when processed
        sqs['command'] : queue containing instructions from master
        sqs['response'] : queue for communication with master
    """
    def __init__(self, directories, init_q ):
        self.name = self._generate_name()
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(static.logging_base_level)
        self.logger.info("Initializing: directories<%s> init_q<%s>" % (json.dumps(directories), init_q) )
        self.s3 = {'source':None, 'results':None}
        self.sqs = {'source':None, 'results':None, 'command': self.name + '-command' , 'response': self.name + '-response' }
        self.directories = directories
        self._terminating = 0
        #counters
        self._hb = 0#heartbeat counter
        self._tcount = 0
        self.ctx = None
        self.em_md5, self.gm_md5, self.sm_md5, self.nm_md5 = (
                None,None,None,None)
        #terminating is state machine
        #see - _terminator for mor info
        def sigterm_handler(*args):
            logger = logging.getLogger("SIGTERM_HANDLER")
            logger.critical("Recd SIGTERM")
            try:
                conn = boto.sqs.connect_to_region( 'us-east-1' )
                command_q = conn.get_queue(self.sqs['command'] )
                parsed = {}
                parsed['message-type'] = 'termination-notice'
                command_q.write(Message(body=json.dumps(parsed)))
                logger.critical("Sending termination notice to command queue")
            except:
                logger.exception("Error during attempted termination")
            #sys.exit()
        signal.signal(signal.SIGTERM, sigterm_handler)
        self._makedirs()
        self._get_settings( init_q )
        try:
            self._init_subprocesses()
        except:
            self.logger.exception("Error on creation of subprocesses, cleanup resources")
            try:
                self.logger.warning("Attempting Hard Cleanup")
                self._hard_clean_up()
            except:
                self.logger.exception("Hard cleanup failed")

    def set_logging_level(self, level):
        self.logger.setLevel(level)

    def run(self):
        """
        The main entry point
        """
        if self._terminating != 0:
            self.logger.critical("Attempted to run, but terminating was set.")
            return
        try:
            self.logger.info("Entering main[run()] process.")
            self._init_gpu()
            self.start_subprocesses()
            self.logger.debug("starting main loop.")
            while self._terminating < 5:
                res = self._main()
                self._heartbeat(force = (not res))
        except:
            self.logger.exception("exception, attempting cleanup" )
            if self._terminating < 5:#otherwise we've already tried this
                try:
                    self._terminator()
                except:
                    self.logger.exception("Terminator failed in exception")
        self.logger.warning("Starting Cleanup")
        try:
            self._heartbeat(True)
        except:
            self.logger.exception("Noone can hear my heart beating, passing error as we are in cleanup")
        try:
            self._hard_clean_up()
        except:
            self.logger.exception("Hard cleanup failed")
        self.logger.info("Exitting main[run()] process.")

    def _main(self):
        """
        This runs the primary logic for gpu
        """
        #get next available data
        #avoiding logging for the main process
        #lean and mean
        db = self._loaderq.next_loader_boss()
        if db is None:
            return False
        self.logger.debug("have data")
        db.clear_data_ready()
        if  db.get_expression_matrix_md5() != self.em_md5:
            expression_matrix = db.get_expression_matrix()
            exp = data.SharedExpression( expression_matrix )
            self.em_md5 = None# db.get_expression_matrix_md5()
            self.exp = exp
        else:
            exp = self.exp
        if  db.get_gene_map_md5() != self.gm_md5:
            gene_map = db.get_gene_map()
            gm = data.SharedGeneMap( gene_map )
            self.gm_md5 = None# db.get_gene_map_md5()
            self.gm= gm
        else:
            gm = self.gm

        if  db.get_sample_map_md5() != self.sm_md5:
            sample_map = db.get_sample_map()
            sm = data.SharedSampleMap( sample_map )
            self.sm_md5 = None # db.get_sample_map_md5()
            self.sm= sm
        else:
            sm = self.sm

        if  db.get_network_map_md5() != self.nm_md5:
            network_map = db.get_network_map()
            nm = data.SharedNetworkMap( network_map )
            self.nm_md5 = None #db.get_network_map_md5()
            self.nm= nm
        else:
            nm = self.nm
        #put in gpu data structures
        #go to work
        srt,rt,rms =  dirac.run( exp, gm, sm, nm, self.sample_block_size, self.pairs_block_size, self.nets_block_size )
        #done with input
        #clear to prevent copying
        exp.clear()
        gm.clear()
        sm.clear()
        nm.clear()
        file_id = db.get_file_id()
        db.release_loader_data()
        db.set_add_data()
        #handle output
        pb = self._packerq.next_packer_boss()
        self.logger.debug("writing to packer")
        rms.fromGPU( pb.get_mem() )
        pb.set_meta(file_id , ( rms.buffer_nnets, rms.buffer_nsamples ))
        pb.release()
        self.logger.debug("<%s> processed and sent to <%s>" %(file_id, pb.name))
        return True

    def _heartbeat(self, force=False):
        """
        Phones home to let master know the status of our gpu node
        """
        if self._hb >= self._hb_interval or force:
            try:
                self.logger.debug("sending heartbeat")
                self._hb = 0
                conn = boto.sqs.connect_to_region( 'us-east-1' )
                response_q = conn.create_queue( self.sqs['response'] )
                mess = self._generate_heartbeat_message()
                response_q.write( mess )
                self._check_commands()
            except:
                self.logger.exception("Heartbeat transmit failed.")
                raise
        else:
            self._hb += 1

    def _terminator(self):
        """
        Handles the logic for shutting down instance.
        """
        #one means soft kill on retriever, so hopefully the pipeline will runout
        self._terminating = 5
        self.logger.warning("Killing Retriever")
        try:
            self._hard_kill_retriever()
        except:
            self.logger.exception("no retrievers to kill")
        self.logger.warning("Killing Loader")
        try:
            self._loaderq.kill_all()
        except:
            self.logger.exception("no loaders to kill")
        self.logger.warning("Killing Packer")
        try:
            self._packerq.kill_all()
        except:
            self.logger.exception("no packers to kill")
        self.logger.warning("Killing Poster")
        try:
            self._hard_kill_poster()
        except:
            self.logger.exception("no posters to kill")
        self.logger.warning("Death to Smoochie")

    def _check_commands(self):
        """
        This checks the command queue to see if any
        instructions from master have arrived.
        TODO: move this into a subprocess
        """
        conn = boto.sqs.connect_to_region( 'us-east-1' )
        command_q = conn.create_queue( self.sqs['command'] )
        for mess in command_q.get_messages(num_messages=10):
            self._handle_command(json.loads(mess.get_body()))
            command_q.delete_message(mess)

    def _handle_command( self, command):
        """
        Given a command from master, initiate change indicated
        """
        if command['message-type'] == 'termination-notice':
            #master says die
            self.logger.warning("received termination notice")
            self._terminating = 1
            self._terminator()
        if command['message-type'] == 'load-balance':
            self.logger.info(str(command))
            self._handle_load_balance(command)
        if command['message-type'] == 'init-settings':
            self.logger.info(str(command))
            self._set_settings( command )

    def _handle_load_balance(self, command):
        """
        Adds or removes subprocesses
        command is structured
        command['message-type'] = 'load-balance'
        command['process'] in ['loader','poster','packer', 'retriever']
        command['type'] in ['add','remove']
        command['increment'] =  integer
        command['min'] = integer !for remove only
        """
        if command['process'] == 'loader':
            self.logger.info("load balancing loader")
            self._lb_loader(command)
        if command['process'] == 'poster':
            self.logger.info("load balancing poster")
            self._lb_poster(command)
        if command['process'] == 'packer':
            self.logger.info("load balancing packer")
            self._lb_packer(command)
        if command['process'] == 'retriever':
            self.logger.info("load balancing retriever")
            self._lb_retriever(command)

    def _lb_loader(self, command):
        """
        Load Balance on Loader
        """
        if command['type'] == 'add':
            self._loaderq.add_loader_boss(num=command['increment'])
        if command['type'] == 'remove':
            num_to_remove = command['increment']
            try:
                for i in range(num_to_remove):
                    if self._loaderq.num_sub() > command['min']:
                        self._loaderq.remove_loader_boss()
            except:
                self.logger.exception("Error on removing loader")
                raise

    def _lb_poster(self,command):
        """
        Load Balance on Poster
        """
        if command['type'] == 'add':
            self._posterq.add_poster(num=command['increment'])
        if command['type'] == 'remove':
            num_to_remove = command['increment']
            try:
                for i in range(num_to_remove):
                    if self._posterq.num_sub > command['min']:
                        self._posterq.remove_poster()
            except:
                self.logger.exception("Error on removing poster")
                raise

    def _lb_packer(self, command):
        """
        Load Balance on Packer
        """
        if command['type'] == 'add':
            self._packerq.add_packer_boss(num=command['increment'])
        if command['type'] == 'remove':
            num_to_remove = command['increment']
            try:
                for i in range(num_to_remove):
                    if self._packerq.num_sub > command['min']:
                        self._packerq.remove_packer_boss()
            except:
                self.logger.exception("Error on removing packer")
                raise

    def _lb_retriever(self, command):
        """
        Load Balance on Retriever
        """
        if command['type'] == 'add':
            self._retrieverq.add_retriever(num=command['increment'])
        if command['type'] == 'remove':
            num_to_remove = command['increment']
            try:
                for i in range(num_to_remove):
                    if self._retrieverq.num_sub > command['min']:
                        self._retrieverq.remove_retriever()
            except:
                self.logger.exception("Error on removing retriever")
                raise

    def _generate_heartbeat_message(self):
        """
        Create a message for master informing current
        state of gpu
        """
        message = self._generate_heartbeat_dict()
        self.logger.info("heartbeat: %s" % json.dumps(message))
        return Message(body=json.dumps(message))

    def _generate_heartbeat_dict(self):
        """
        Creates the dictionary holding state information for heartbeat
        """
        message = {}
        message['message-type'] = 'gpu-heartbeat'
        try:
            message['name'] = self.name
            message['num-packer'] = self._packerq.num_sub()
            message['num-poster'] = self._posterq.num_sub()
            message['num-retriever'] = self._retrieverq.num_sub()
            message['num-loader'] = self._loaderq.num_sub()
            message['source-q'] = self._source_q.qsize()
            message['result-q'] = self._result_q.qsize()
            message['terminating'] = self._terminating
            message['time'] = time.time()
        except:
            self.logger.exception("Heartbeat message generation error")
            raise
        return message

    def _init_gpu(self):
        """
        Initialize gpu context
        """
        self.logger.info("starting cuda")
        cuda.init()
        dev = cuda.Device( self.gpu_id )
        self.ctx = dev.make_context()

    def _catch_cuda(self):
        """
        In case of an uncaught, unrecoverable exception
        pop the gpu context
        """
        if self.ctx is not None:
            try:
                self.logger.info("killing cuda")
                self.ctx.pop()
            except:
                self.logger.error("unable to successfully clear context")

    def _get_settings(self, init_q_name):
        """
        Alert master to existence, via sqs with init_q_name
        Get initial settings
        """
        conn = boto.sqs.connect_to_region( 'us-east-1' )
        init_q = None
        ctr = 0
        self._generate_command_queues()
        while init_q is None and ctr < 6:
            init_q = conn.get_queue( init_q_name  )
            time.sleep(1+ctr**2)
            ctr += 1
        if init_q is None:
            self.logger.error("Unable to connect to init q")
            raise Exception("Unable to connect to init q")
        md =  boto.utils.get_instance_metadata()
        self._availabilityzone = md['placement']['availability-zone']
        self._region = self._availabilityzone[:-1]
        message = {'message-type':'gpu-init',
                'name': self.name,
                'cluster-name': self.get_cluster_name(),
                'instance-id': md['instance-id'],
                'command' : self.sqs['command'],
                'response' : self.sqs['response'],
                'zone':self._availabilityzone }
        m = Message(body=json.dumps(message))
        init_q.write( m )
        command_q = conn.get_queue( self.sqs['command'] )
        command = None
        ctr = 0
        while command is None:
            command = command_q.read(  wait_time_seconds=20 )
            if command is None:
                self.logger.warning("No instructions in [%s]"%self.sqs['command'])
            ctr += 1
        if command is None:
            self.logger.error("Attempted to retrieve setup and no instructions received.")
            raise Exception("Waited 200 seconds and no instructions, exitting.")
        self.logger.debug("Init Message %s", command.get_body())
        parsed = json.loads(command.get_body())
        command_q.delete_message( command )
        self._handle_command(parsed)
        try:
            self.logger.debug("sqs< %s > s3< %s > ds< %s > gpu_id< %s >" % (str(self.sqs), str(self.s3), str(self.data_settings), str(self.gpu_id)) )
        except AttributeError:
            self.logger.exception("Probably terminated before initialization")

    def get_cluster_name( self ):
        return '-'.join(socket.gethostname().split('-')[:-1])

    def _set_settings( self, command):
        """
        Given a command dictionary containing a global config,
        set instance variables necessary for startup.
        """
        self.sqs['results'] = command['result-sqs']
        self.sqs['source'] = command['source-sqs']
        self.s3['source'] = command['source-s3']
        self.s3['results'] = command['result-s3']
        self.data_settings = self._reformat_data_settings(command['data-settings'])
        self.gpu_id = command['gpu-id']
        self.sample_block_size = command['sample-block-size']
        self.pairs_block_size = command['pairs-block-size']
        self.nets_block_size = command['nets-block-size']
        self._hb_interval = command['heartbeat-interval']

    def _reformat_data_settings(self, data_settings):
        new_data_settings = {}
        for k in data_settings.iterkeys():
            new_data_settings[k] = []
            for dt, size, dtype in data_settings[k]:
                self.logger.debug("data_settings[%s]: (%s, %i, %s )" %(k, dt, size, dtypes.nd_list[dtype]))
                new_data_settings[k].append( (dt, size, dtypes.nd_list[dtype]) )
        return new_data_settings

    def _generate_command_queues(self):
        """
        Create the command queues for this process
        Command Queues are queues that are used to communicate
        status and instructions between this process and the cluster.
        """
        conn = boto.sqs.connect_to_region( 'us-east-1' )
        response_q = conn.create_queue( self.sqs['response'] )
        command_q = conn.create_queue( self.sqs['command'] )
        #check that queue was actually created
        command_q = None
        while command_q is None:
            command_q = conn.get_queue( self.sqs['command'] )
            time.sleep(1)
        response_q = None
        while response_q is None:
            response_q = conn.get_queue( self.sqs['response'] )
            time.sleep(1)

    def _delete_command_queues(self):
        """
        Command queues are created by and specific to this process,
        clean them up when done.
        """
        conn = boto.sqs.connect_to_region( 'us-east-1' )
        command_q = conn.get_queue( self.sqs['command'] )
        if command_q is not None:
            self.logger.warning("Deleting [%s]" %  self.sqs['command'])
            command_q.delete()
        response_q = conn.get_queue( self.sqs['response'] )
        if response_q is not None:
            ctr = 0
            while response_q.count() > 0 and ctr < 10:
                self.logger.warning("Trying to delete queue, but have unread \
                    messages in response queue.")
                time.sleep(1)
                ctr += 1
            if response_q.count():
                dump_path = os.path.join(self.directories['log'],
                                self.name + "-response-queue-unsent.log")
                self.logger.warning("Dumping response queue to [%s]" % (dump_path,)    )
                response_q.dump(dump_path, sep='\n\n')
            self.logger.warning( "Deleting [%s]" % self.sqs['response'] )
            response_q.delete()

    def _generate_name(self):
        """
        Create a unique name for this process
        """
        md =  boto.utils.get_instance_metadata()
        pid = str( multiprocessing.current_process().pid )
        return md['instance-id'] + '_' + pid

    def _init_subprocesses(self):
        """
        Initializes (but does not start) worker processes.
        """
        self.logger.debug("Initializing subprocesses")
        self._source_q = Queue()#queue containing names of source data files for processing
        self._result_q = Queue()#queue containing names of result data files from processing
        self._retrieverq = RetrieverQueue( self.name + "_RetrieverQueue",
                    self.directories['source'], self._source_q,
                    self.sqs['source'], self.s3['source'] )
        self._posterq = PosterQueue( self.name + "_PosterQueue",
                    self.directories['results'], self._result_q,
                    self.sqs['results'], self.s3['results'],
                    self.directories['source'], self.sqs['source'] )
        self._loaderq = LoaderQueue( self.name + "_LoaderQueue",
                    self._source_q, self.directories['source'],
                    data_settings = self.data_settings['source'] )
        self._packerq = PackerQueue( self.name + "_PackerQueue",
                    self._result_q, self.directories['results'],
                    data_settings = self.data_settings['results'] )
        self.logger.debug("Subprocesses Initialized" )

    def start_subprocesses(self):
        """
        Starts subprocesses
        """
        self.logger.debug("Starting subprocesses")
        self._retrieverq.add_retriever(5)
        self._posterq.add_poster(5)
        self._loaderq.add_loader_boss(5)
        self._packerq.add_packer_boss(5)

    def _hard_kill_retriever(self):
        """
        Terminates retriever subprocesses
        """
        self.logger.warning("Hard Kill Retriever")
        self._retrieverq.kill_all()
        ctr = 0
        while not self._retrieverq.all_dead() and ctr < 5:
            self.logger.warning("Retriever not dead yet")
            time.sleep(1)
            ctr += 1
        self._retrieverq.clean_up()

    def _hard_kill_poster(self):
        """
        Terminates poster subprocesses.
        May be unuploaded files.
        """
        self._posterq.kill_all()
        #sleeps in posterqueue
        self._posterq.clean_up()

    def _hard_clean_up(self):
        """
        This cleans up anything that did not end gracefully
        """
        if self._terminating == 0:
            self._terminating = 5
        self.logger.info("Hard Cleanup routine")
        for c in multiprocessing.active_children():
            self.logger.warning("Hard kill [%s]" % c.name)
            c.terminate()
        self._catch_cuda()

    def _makedirs(self):
        """
        Creates directories listed in directories
        If they do not exist
        """
        error = True
        ctr = 0
        while error:
            error = False
            ctr += 1
            for k, p in self.directories.iteritems():
                if not os.path.exists(p):
                    try:
                        os.makedirs(p)
                    except:
                        self.logger.error("tried to make directory [%s], failed." %p )
                        #might have multiple procs trying this, if already done, ignore
                        error = True
                        if ctr >= 10:
                            self.logger.error("failed to create directory too many times." )
                            raise

def main():
    import masterdirac.models.systemdefaults as sys_def
    import os, os.path
    directories = {}
    directories = sys_def.get_system_defaults(component='Dual GPU',
            setting_name='directories')
    q_cfg = sys_def.get_system_defaults(component='Dual GPU',
            setting_name='queues')
    init_q = q_cfg['init_q']
    debug.initLogging()
    d = Dirac( directories, init_q )
    d.run()

if __name__ == "__main__":
    #debug.initLogging()
    print "Nothing Here, look at main()"
