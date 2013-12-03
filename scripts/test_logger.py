import logging
import gpudirac.utils.debug as db

db.initLogging()
logger = logging.getLogger('test')
logger.debug("Testing Logger")
logger.warning("test logger")
logger.error("Test Logger")
logger.critical("Test Logger")


