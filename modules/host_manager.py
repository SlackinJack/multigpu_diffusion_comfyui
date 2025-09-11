import base64
import errno
import json
import os
import psutil
import requests
import signal
import socket
import subprocess
import threading
import time
import torch
from torchvision.transforms import ToPILImage, ToTensor


from ..multigpu_diffusion.modules.utils import *
from ..nodes.globals import *
from ..nodes.nodes_general import *


config                  = json.load(open(os.path.dirname(os.path.abspath(__file__)) + "/../multigpu_diffusion/config.json"))
config_global           = config["global"]
PORT                    = str(config_global["port"])
MASTER_PORT             = str(config_global["master_port"])
host_address            = f'http://localhost:{PORT}'
host_address_generate   = f'{host_address}/generate'
host_address_initialize = f'{host_address}/initialize'
host_address_progress   = f'{host_address}/progress'


host_process            = None
host_process_backend    = ""
last_config             = None


def launch_host(config, backend, bar):
    global cwd, MASTER_PORT, PORT, host_process, host_process_backend, last_config
    closed = False
    if host_process_backend != backend:
        close_host_process("Backend changed")
        closed = True
    if not closed and (host_process is not None and last_config is not None):
        for k, v in last_config.items():
            if k not in config:
                close_host_process("Updated configuration")
                closed = True
                break
            elif config[k] != v:
                close_host_process("Updated configuration")
                closed = True
                break
    if not closed and (host_process is None or last_config is None or len(host_process_backend) == 0):
        close_host_process("Host not set correctly")
        closed = True

    if not closed:
        bar.update_absolute(33)
    else:
        nproc_per_node = config.pop("nproc_per_node")
        cmd = [
            "torchrun",
            f'--nproc_per_node={nproc_per_node}',
            f'--master-port={MASTER_PORT}',
            f'{cwd}/../multigpu_diffusion/host_{backend}.py',
            f'--port={PORT}'
        ]

        for k, v in config.items():
            if k in GENERIC_CONFIGS_COMFY.keys(): continue
            if k in ["lora", "control_net", "ip_adapter", "motion_adapter_lora"]:
                cmd.append(f'--{k}={json.dumps(v)}')
            else:
                v = str(v)
                if v == "True":
                    cmd.append(f'--{k}')
                else:
                    if v == "False": continue
                    if k == "quantize_to" and v == "disabled": continue
                    cmd.append(f'--{k}={str(v)}')

        # launch host
        cmd_str = ""
        for c in cmd: cmd_str += '\n' + c
        print(f'Starting host:{str(cmd_str)}')
        host_process = subprocess.Popen(cmd)
        last_config = config
        host_process_backend = backend
        check_host_process(config.get("pipeline_init_timeout"), bar)
        bar.update_absolute(33)
    return


def check_host_process(timeout, bar):
    global host_address_initialize
    current = 0
    while True:
        try:
            response = requests.get(host_address_initialize)
            if response.status_code == 200:
                bar.update_absolute(33)
                return
        except requests.exceptions.RequestException:
            pass
        bar.update_absolute(current / timeout * 33)
        time.sleep(1)
        current += 1
        if current >= timeout:
            close_host_process("Timed out")
            assert False, f'Failed to launch host within {timeout} seconds.\nCheck console for details.'


def close_host_process(reason):
    global host_process, last_config, host_process_backend, PORT
    if host_process is not None:
        print(f'Stopping host process: {reason}')
        host = psutil.Process(host_process.pid)
        workers = [host] + host.children(recursive=True)
        while True:
            for w in workers:
                try:
                    print(f'Stopping PID: {w.pid}')
                    w.terminate()
                    time.sleep(3)
                    w.kill()    
                except:
                    pass
            try:
                time.sleep(3)
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("localhost", int(PORT)))
                time.sleep(3)
                s.close()
                time.sleep(1)
                host_process = None
                last_config = None
                host_process_backend = ""
                print('Host has been stopped')
                return
            except socket.error as e:
                if e.errno == errno.EADDRINUSE:
                    print('Host still active - waiting to retry')
                    time.sleep(3)
            except Exception as ex:
                print(f'Error occurred: {str(ex)} - waiting to retry')
                time.sleep(3)


def get_result(data, bar):
    global host_address_generate, host_address_progress, last_config
    run = True

    def get_progress():
        while True:
            nonlocal bar, run
            if run:
                try:
                    progress = requests.get(host_address_progress)
                    if progress is not None and progress.text is not None:
                        bar.update_absolute(33 + (float(progress.text) / 100 * 67))
                except Exception as e:
                    pass
                time.sleep(5)
            else:
                return

    prog_thread = threading.Thread(target=get_progress)
    prog_thread.start()

    try:
        response = requests.post(host_address_generate, json=data)
    except Exception as e:
        assert False, f"Could not post request to host.\nCheck console for details.\n\n{str(e)}"

    run = False
    bar.update_absolute(100)

    try:
        response_data = response.json()
        output_base64 = response_data.get("output")
        if output_base64 is None:
            assert response_data is not None, "No response from host.\nCheck console for details."
            assert False, response_data.get("message")
        if last_config is not None and not last_config["keepalive"]:
            close_host_process("Keepalive option off")
        return output_base64
    except Exception as e:
        run = False
        close_host_process("Unknown error")
        assert False, f"An error occurred while generating image.\nCheck console for details.\n\n{str(e)}"

