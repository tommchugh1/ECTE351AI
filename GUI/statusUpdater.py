import threading
import socket

class StatusMonitor:
    """
    Periodically checks stream status in a background thread, then calls `on_status`
    back on the Tk thread. Non-blocking, stoppable, and you can tweak the interval.
    Status strings are: "running" | "starting" | "stopped".
    """
    def __init__(self, tk_root, check_fn, on_status, interval_ms: int = 800):
        self.root = tk_root
        self.check_fn = check_fn          # callable() -> str
        self.on_status = on_status        # callable(status: str)
        self.interval_ms = max(50, int(interval_ms))
        self._stopped = False
        self._tick()  # start the loop

    def _tick(self):
        if self._stopped:
            return

        def worker():
            try:
                status = self.check_fn()
            except Exception:
                status = "stopped"
            # marshal back to Tk thread
            if not self._stopped:
                self.root.after(0, lambda: None if self._stopped else self.on_status(status))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(self.interval_ms, self._tick)

    def stop(self):
        self._stopped = True

    def set_interval(self, ms: int):
        self.interval_ms = max(50, int(ms))


# Lightweight TCP probe helpers 
def is_tcp_open(host: str, port: int, timeout: float = 0.08) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_status_rtsp_port(host: str, port: int = 8554) -> str:
    return "running" if is_tcp_open(host, port) else "stopped"


#Temporarily increase polling frequency for snappier feedback after Start/Stop.
def nudge_monitor_fast(monitor: StatusMonitor, ms: int = 250, for_seconds: float = 3.0):
    if not monitor:
        return
    old = monitor.interval_ms
    monitor.set_interval(ms)
    monitor.root.after(int(for_seconds * 1000), lambda: monitor.set_interval(old))
