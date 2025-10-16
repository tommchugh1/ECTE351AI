import socket
import subprocess
import threading
import os
import signal
import time
from pathlib import Path

working_dir = os.path.dirname(os.path.abspath(__file__))
HOST = "0.0.0.0"   
PORT = 9001        
MEDIA_MTX_PATH = os.path.join(working_dir, "mediamtx", "mediamtx")  
CONFIG_FILE = os.path.join(working_dir, "mediamtx.yml")

active_process = None
lock = threading.Lock()

# Helpers

def _proc_is_running_nolock() -> bool:
    # Internal: assumes caller holds the lock
    global active_process
    return bool(active_process and (active_process.poll() is None))

def is_mediamtx_running() -> bool:
    # Thread safe running check
    with lock:
        return _proc_is_running_nolock

def start_mediamtx():
    # Start MediaMTX server non-blocking and return immediately
    global active_process
    with lock:
        if _proc_is_running_nolock():
            print("[PI] start_mediamtx: already running")
            return "OK: already running"

        bin_path = Path(MEDIA_MTX_PATH)
        cfg_path = Path(CONFIG_FILE)

        if not bin_path.exists():
            print(f"[PI][ERROR] Binary not found: {bin_path}")
            return f"ERROR: binary not found ({bin_path})"
        if not cfg_path.exists():
            print(f"[PI][WARN] Config not found: {cfg_path} (will try to run without)")

        try:
            # Build args
            args = [str(bin_path)]
            if cfg_path.exists():
                args += [str(cfg_path)]

            print(f"[PI] Starting MediaMTX: {' '.join(args)}")
            active_process = subprocess.Popen(
                args,
                cwd=str(bin_path.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid  # new process group
            )
            # Do NOT block here, let the GUI poll with RTSP DESCRIBE
            return "OK: starting"
        except Exception as e:
            print(f"[PI][ERROR] Failed to start MediaMTX: {e}")
            return f"ERROR: {e}"

def stop_mediamtx():
    # Stop MediaMTX process if running.
    global active_process
    with lock:
        if not _proc_is_running_nolock():
            print("[PI] MediaMTX is not running.")
            active_process = None
            return "Not running"
        try:
            print("[PI] Stopping MediaMTX (SIGTERM)...")
            os.killpg(os.getpgid(active_process.pid), signal.SIGTERM)
            # Wait up to 4s
            deadline = time.time() + 4.0
            while time.time() < deadline:
                if active_process.poll() is not None:
                    print("[PI] MediaMTX stopped cleanly.")
                    active_process = None
                    return "OK: stopped"
                time.sleep(0.1)

            print("[PI] Forcing kill (SIGKILL)...")
            os.killpg(os.getpgid(active_process.pid), signal.SIGKILL)
            active_process = None
            return "OK: killed"
        except Exception as e:
            print(f"[PI][ERROR] Failed to stop MediaMTX: {e}")
            return f"ERROR: {e}"

# Command Handler

def handle_client(conn, addr):
    """Handle incoming single-line commands."""
    try:
        conn.settimeout(2.0)
        data = conn.recv(1024)
        cmd = (data or b"").decode("utf-8", errors="ignore").strip()
        print(f"[PI] Command from {addr}: {cmd!r}")

        if cmd == "start_feed":
            # Start immediately and ACK right away
            result = start_mediamtx()
        elif cmd == "stop_feed":
            result = stop_mediamtx()
        elif cmd == "status":
            result = "running" if is_mediamtx_running() else "stopped"
        else:
            result = f"ERROR: unknown command '{cmd}'"

        # Always try to send a response (non-blocking operations above)
        try:
            conn.sendall(result.encode("utf-8"))
        except Exception as e:
            print(f"[PI][WARN] sendall failed: {e}")

    except socket.timeout:
        print(f"[PI][WARN] Timeout receiving data from {addr}")
        try:
            conn.sendall(b"ERROR: timeout")
        except Exception:
            pass
    except Exception as e:
        print(f"[PI][ERROR] Handler error: {e}")
        try:
            conn.sendall(f"ERROR: {e}".encode("utf-8"))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

# Server Loop

def main():
    print(f"[PI] Camera control server listening on {HOST}:{PORT}")
    # Pre-flight: print paths to help debugging
    print(f"[PI] MediaMTX: {MEDIA_MTX_PATH}")
    print(f"[PI] Config  : {CONFIG_FILE}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(8)
        print("[PI] Ready (commands: start_feed | stop_feed | status)")

        try:
            while True:
                conn, addr = server.accept()
                t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
                t.start()
        except KeyboardInterrupt:
            print("\n[PI] KeyboardInterrupt: shutting down.")
        finally:
            # Best effort stop on exit
            try:
                stop_mediamtx()
            except Exception:
                pass

if __name__ == "__main__":
    main()
