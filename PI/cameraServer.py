import socket
import subprocess
import threading
import os
import signal
import time

working_dir = os.path.dirname(os.path.abspath(__file__))
HOST = "0.0.0.0"   
PORT = 9001        
MEDIA_MTX_PATH = os.path.join(working_dir, "mediamtx")  
CONFIG_FILE = os.path.join(working_dir, "mediamtx.yml")

active_process = None
lock = threading.Lock()

# Helpers

def is_mediamtx_running() -> bool:
    global active_process
    with lock:
        if active_process and active_process.poll() is None:
            return True
    return False

def start_mediamtx():
    # Start the MediaMTX server with config file.
    global active_process
    with lock:
        if is_mediamtx_running():
            return "Already running"
        try:
            print("[PI] Starting MediaMTX...")
            active_process = subprocess.Popen(
                [MEDIA_MTX_PATH, "-config", CONFIG_FILE],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid  # Start new process group
            )
            time.sleep(2)  # Allow process to initialize
            if is_mediamtx_running():
                print("[PI] MediaMTX started successfully.")
                return "OK"
            else:
                print("[PI] Failed to start MediaMTX.")
                return "FAIL"
        except Exception as e:
            print(f"[ERROR] Failed to start MediaMTX: {e}")
            return f"ERROR: {e}"

def stop_mediamtx():
    # Stop MediaMTX process if running.
    global active_process
    with lock:
        if not is_mediamtx_running():
            print("[PI] MediaMTX is not running.")
            return "Not running"
        try:
            print("[PI] Stopping MediaMTX...")
            os.killpg(os.getpgid(active_process.pid), signal.SIGTERM)
            active_process.wait(timeout=5)
            active_process = None
            print("[PI] MediaMTX stopped successfully.")
            return "OK"
        except Exception as e:
            print(f"[ERROR] Failed to stop MediaMTX: {e}")
            return f"ERROR: {e}"


# Command Handler

def handle_client(conn, addr):
    # Handle incoming command from the NUC.
    try:
        data = conn.recv(1024).decode("utf-8").strip()
        print(f"[PI] Command received from {addr}: {data}")

        if data == "start_feed":
            response = start_mediamtx()
        elif data == "stop_feed":
            response = stop_mediamtx()
        elif data == "status":
            response = "running" if is_mediamtx_running() else "stopped"
        else:
            response = f"Unknown command: {data}"

        conn.sendall(response.encode("utf-8"))
    except Exception as e:
        print(f"[ERROR] Client handler exception: {e}")
        try:
            conn.sendall(f"ERROR: {e}".encode("utf-8"))
        except Exception:
            pass
    finally:
        conn.close()

# Server Loop

def main():
    print(f"[PI] Camera control server starting on port {PORT}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(5)
        print("[PI] Ready to receive commands (start_feed / stop_feed / status).")

        while True:
            try:
                conn, addr = server.accept()
                threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
            except KeyboardInterrupt:
                print("\n[PI] Shutting down server.")
                stop_mediamtx()
                break
            except Exception as e:
                print(f"[ERROR] Server exception: {e}")
                time.sleep(1)

if __name__ == "__main__":
    main()
