import logging
import gpudirac.utils.debug as db
import time

db.initLogging(configfile='../config/config.cfg')
ctr = 0
while True:
    ctr += 1
    logger = logging.getLogger('test_' + str(ctr))
    logger.debug("Testing Logger")
    logger.warning("test logger")
    logger.error("Test Logger")
    logger.critical("Test Logger")
    time.sleep(1)


