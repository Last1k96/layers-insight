from queue import Queue
import threading
import diskcache as dc

# Shared resources
result_cache = {}
task_queue = Queue()
lock = threading.Lock()

cache = dc.Cache("./cache")
