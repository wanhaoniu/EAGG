import sys
import time

class Logger:
    """
    Simple logger to print messages to console (and optionally to a file).
    """
    def __init__(self, log_file=None):
        self.log_file = None
        if log_file:
            self.log_file = open(log_file, "a")
    
    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        print(msg)
        if self.log_file:
            self.log_file.write(msg + "\n")
            self.log_file.flush()
    
    def close(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None
