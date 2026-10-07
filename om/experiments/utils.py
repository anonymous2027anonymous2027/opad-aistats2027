import time

class TimeKeeper:
    def __init__(self):
        self.prev_total_passed_time = 0  # accumulated
        self.current_start_time = None

    def is_active(self):
        return self.current_start_time is not None
    def start(self):
        self.current_start_time = time.perf_counter()

    def stop(self):
        assert self.is_active(), "not started to stop"
        self.prev_total_passed_time += (time.perf_counter() - self.current_start_time)
        self.current_start_time = None  # become passive

    def get_passed_time(self):
        if self.is_active():
            return self.prev_total_passed_time + (time.perf_counter() - self.current_start_time)
        else:
            return self.prev_total_passed_time
