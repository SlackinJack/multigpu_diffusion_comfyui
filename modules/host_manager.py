import base64
import copy
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


configs = {}
#{
#    "port": {
#        "host_process": host_process,
#        "loaded_backend": backend,
#        "last_config": last_config,
#    }
#}


def launch_host(config, backend, bar):
    global cwd, configs
    closed = False

    port = config.get("port")
    current_config = configs.get(str(port))
    if current_config is None:
        closed = True
    else:
        host_process = current_config.get("host_process")
        loaded_backend = current_config.get("loaded_backend")
        last_config = current_config.get("last_config")

        if host_process is None or loaded_backend is None or last_config is None:
            close_host_process(port, host_process, "Invalid host configuration", wait_for_close=True)
            closed = True

        if not closed and (loaded_backend != backend or len(loaded_backend) == 0):
            close_host_process(port, host_process, "Backend changed", wait_for_close=True)
            closed = True

        if not closed:
            for k, v in last_config.items():
                if k not in config:
                    close_host_process(port, host_process, "Updated configuration", wait_for_close=True)
                    closed = True
                    break
                if str(config[k]) != str(v):
                    close_host_process(port, host_process, "Updated configuration", wait_for_close=True)
                    closed = True
                    break

    if not closed:
        bar.update_absolute(33)
    else:
        cmd = [
            "torchrun",
            f'--nproc_per_node={config["nproc_per_node"]}',
            f'--master-port={config["master_port"]}',
            f'{cwd}/../multigpu_diffusion/host_{backend}.py',
            f'--port={config["port"]}'
        ]

        for k, v in config.items():
            if k in ["nproc_per_node", "master_port", "port"]: continue
            if k in GENERIC_CONFIGS_COMFY.keys(): continue
            if k in ["scheduler", "lora", "control_net", "ip_adapter", "motion_adapter_lora"]:
                cmd.append(f'--{k}={json.dumps(v)}')
            else:
                v = str(v)
                if v == "True":
                    cmd.append(f'--{k}')
                else:
                    if v == "False":                                                                                continue
                    if k in ["quantize_unet_to", "quantize_encoder_to", "quantize_misc_to"] and v == "disabled":    continue
                    if k in ["compile_backend", "compile_mode"] and v == "default":                                 continue
                    if k in ["compile_options"] and len(v) == 0:                                                    continue
                    cmd.append(f'--{k}={str(v)}')

        # launch host
        cmd_str = ""
        for c in cmd: cmd_str += '\n' + c
        print(f'Starting host:{str(cmd_str)}')
        host_process = subprocess.Popen(cmd)
        configs[str(port)] = {
            "host_process": host_process,
            "loaded_backend": backend,
            "last_config": config,
        }
        timeout = config.get("pipeline_init_timeout")
        check_host_process(host_process, port, timeout, bar)
        bar.update_absolute(33)
    return


def check_host_process(host_process, port, timeout, bar):
    global configs
    current = 0
    while True:
        try:
            response = requests.get(f"http://localhost:{port}/initialize")
            if response.status_code == 200:
                bar.update_absolute(33)
                return
        except requests.exceptions.RequestException:
            pass
        bar.update_absolute(current / timeout * 33)
        time.sleep(1)
        current += 1
        if current >= timeout:
            close_host_process(port, host_process, "Timed out", wait_for_close=True)
            assert False, f'Failed to launch host within {timeout} seconds.\nCheck console for details.'


def close_host_process(port, host_process, reason, wait_for_close=False):
    global configs
    if host_process is not None:
        if wait_for_close:  print(f'Synchronously stopping host process ({reason})')
        else:               print(f'Stopping host process ({reason})')
        def close(port, host_process):
            host = psutil.Process(host_process.pid)
            workers = [host] + host.children(recursive=True)
            while True:
                for w in workers:
                    result = subprocess.run(["kill", "-9", str(w.pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    time.sleep(3)
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.bind(("localhost", port))
                    time.sleep(1)
                    s.close()
                    if configs.get(str(port)) is not None: del configs[str(port)]
                    print('Host has been stopped')
                    return
                except socket.error as e:
                    if e.errno == errno.EADDRINUSE:
                        print('Host still active - waiting to retry')
                        time.sleep(3)
                except Exception as ex:
                    print(f'Error occurred - waiting to retry\n{str(ex)}')
                    time.sleep(3)
        if not wait_for_close:  threading.Thread(target=close, args=(port, host_process,)).start()
        else:                   close(port, host_process)
    return


def get_result(port, data, bar):
    global configs
    assert configs.get(str(port)) is not None, "Host not initialized."
    host_process = configs[str(port)]["host_process"]
    last_config = configs[str(port)]["last_config"]
    run = True

    def get_progress():
        while True:
            nonlocal bar, run
            if run:
                try:
                    progress = requests.get(f"http://localhost:{port}/progress")
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
        response = requests.post(f"http://localhost:{port}/generate", json=data)
    except Exception as e:
        close_host_process(port, host_process, "Connection error", wait_for_close=True)
        assert False, f"Could not post request to host.\nCheck console for details.\n\n{str(e)}"

    run = False
    bar.update_absolute(100)

    try:
        response_data = response.json()
        output_image_b64 = response_data.get("output")
        output_latent_b64 = response_data.get("latent")
        if last_config is not None and not last_config["keepalive"]:
            close_host_process(port, host_process, "Keepalive option off", wait_for_close=last_config["wait_for_host_close"])
        if output_image_b64 is None and output_latent_b64 is None:
            assert response_data is not None, "No response from host.\nCheck console for details."
            assert False, response_data.get("message")
        else:
            if output_latent_b64 is not None:   return output_image_b64, output_latent_b64
            else:                               return output_image_b64
    except Exception as e:
        run = False
        close_host_process(port, host_process, "Unknown error", wait_for_close=True)
        assert False, f"An error occurred while generating image.\nCheck console for details.\n\n{str(e)}"

