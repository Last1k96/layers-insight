from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor

# Shared resources
result_cache = {}
task_queue = Queue()
lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)