DEBUG = True
import time
import os
import os.path
import sys

import pickle
import logging
import logging.handlers
import SocketServer
import struct
import socket
from multiprocessing import Process, Event

import static
import boto
import boto.utils
from boto.s3.key import Key

class TimeTracker:
    def __init__(self):
        self._wait_tick = time.time()
        self._work_tick = time.time()
        self._waiting = 0.0
        self._working = 0.0

    def start_work(self):
        self._work_tick= time.time()

    def end_work(self):
        self._working += time.time() - self._work_tick
        self._work_tick = time.time()


    def start_wait(self):
        self._wait_tick= time.time()

    def end_wait(self):
        self._waiting += time.time() - self._wait_tick
        self._wait_tick = time.time()

    def print_stats(self):
        print
        print "Waiting Time:", self._waiting
        print "Working Time:", self._working
        print "working/waiting", self._working/self._waiting
        print
"""
Below is lightly modified from http://docs.python.org/2/howto/logging-cookbook.html#logging-cookbook
"""

class LogRecordStreamHandler(SocketServer.StreamRequestHandler):
    """Handler for a streaming logging request.

    This basically logs the record using whatever logging policy is
    configured locally.
    """
    def handle(self):
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = self.unPickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unPickle(self, data):
        return pickle.loads(data)

    def handleLogRecord(self, record):
        # if a name is specified, we use the named logger rather than the one
        # implied by the record.
        if self.server.logname is not None:
            name = self.server.logname
        else:
            name = record.name
        logger = logging.getLogger(name)
        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        logger.handle(record)

class LogRecordSocketReceiver(SocketServer.ThreadingTCPServer):
    """
    Simple TCP socket-based logging receiver suitable for testing.
    """
    allow_reuse_address = 1
    def __init__(self, host='localhost', port=logging.handlers.DEFAULT_TCP_LOGGING_PORT, handler=LogRecordStreamHandler):
        SocketServer.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.abort = 0
        self.timeout = 1
        self.logname = None

    def serve_until_stopped(self):
        import select
        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()], [], [],  self.timeout)
            abort = self.abort
            if rd:
                self.handle_request()

class S3TimedRotatatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, when='h', interval=1, backupCount=0, encoding=None, delay=False, utc=False, bucket):
        TimedRotatingFileHandler. __init__(self, filename, when, interval, backupCount, encoding, delay, utc)
        self.bucket = bucket

   def doRollover(self):
        """
        do a rollover; in this case, a date/time stamp is appended to the filename
        when the rollover happens.  However, you want the file to be named for the
        start of the interval, not the current time.  If there is a backup count,
        then we have to get a list of matching filenames, sort them and remove
        the one with the oldest suffix.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)
        if os.path.exists(dfn):
            os.remove(dfn)
        # Issue 18940: A file may not have been created if delay is True.
        if os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, dfn)
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        if not self.delay:
            self.stream = self._open()
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        #If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:           # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt
        #this is the new stuff
        #copy to s3 the old file
        self.pushToS3( dfn )

    def pushToS3(self, filename):
        conn = boto.connect_s3()
        bucket = conn.get_bucket(self.bucket)
        k = Key(bucket)
        k.key = os.path.split(filename)[1]
        k.set_contents_from_filename(filename, reduced_redundancy=True)

def startLogger():
    import argparse
    import ConfigParser
    import os, os.path
    import logging 
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Configfile name', required=True)
    args = parser.parse_args()
    config = ConfigParser.ConfigParser(interpolation=None)
    config.read(args.config)
    
    log_dir = config.get('base', 'directory')
    LOG_FILENAME = config.get('base', 'log_filename')
    if LOG_FILENAME is 'None':
        md =  boto.utils.get_instance_metadata()
        LOG_FILENAME = md['instance-id'] + '.log'
    log_format = config.get('base', 'log_format')
    bucket = config.get('base', 'bucket')
    interval_type = config.get('base', 'interval_type')
    interval = int(config.get('base', 'interval'))
    logging.basicConfig(format=log_format)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    handler = logging.handlers.RotatingFileHandler()#log 100 MB
    handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger('').addHandler(handler)
    tcpserver = LogRecordSocketReceiver()
    print('About to start TCP server...')
    tcpserver.serve_until_stopped()

def initLogging(server='localhost', port=logging.handlers.DEFAULT_TCP_LOGGING_PORT, server_level=static.logging_server_level, sys_out_level=static.logging_stdout_level):
    rootLogger = logging.getLogger('')
    rootLogger.setLevel(logging.DEBUG)
    botoLogger = logging.getLogger('boto')
    botoLogger.setLevel(logging.WARNING)
    socketHandler = logging.handlers.SocketHandler(server, port)
    socketHandler.setLevel(server_level)
    rootLogger.addHandler(socketHandler)
    if sys_out_level is not None:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(sys_out_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        rootLogger.addHandler(ch)

if __name__ == "__main__":
        initLogging()
        logging.error("test")
        logger = logging.getLogger("test1-new")
        logger.error("test")

