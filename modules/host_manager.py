import base64
import copy
import errno
import gc
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
from ..nodes.data_types import *


LOCAL_HOST = "http://localhost:"


class HostManager:
    configs = {}
    #{
    #    "port": {
    #        "process": process,
    #        "backend": backend,
    #        "last_config": last_config,
    #        "pipeline": pipeline,
    #        "closed": BOOLEAN,
    #    }
    #}


    def __get_config_by_address(self, address):
        port = address.replace(LOCAL_HOST, "")
        c = self.configs.get(port)
        if c is not None:
            if c["closed"] == True and c["pipeline"] is not None:
                pipeline_to_load = copy.deepcopy(c["pipeline"])
                if c.get("last_config") is None:
                    self.close_host_process(address, "Misconfigured host", wait_for_close=True, with_assert="Misconfigured host.")
                self.launch_host(c["last_config"])
                self.post_to_address(LOCAL_HOST+port, "apply", pipeline_to_load)
        return self.configs[port]


    def __set_config_for_address(self, address, config):
        if LOCAL_HOST in address: address = address.replace(LOCAL_HOST, "")
        self.configs[address] = config
        return


    def __update_config_pipeline(self, address, value):
        config = self.__get_config_by_address(address)
        config["pipeline"] = value
        return


    def get_from_address(self, address, endpoint):
        if not LOCAL_HOST in address: address = LOCAL_HOST + address
        result = requests.get(f"{address}/{endpoint}")

        # clean up after request
        torch.cuda.memory.empty_cache()
        gc.collect()

        return result


    def post_to_address(self, address, endpoint, data):
        if not LOCAL_HOST in address: address = LOCAL_HOST + address
        match endpoint:
            case "apply":   self.__update_config_pipeline(address, data)
            case _:         pass

        result = None
        try:                result = requests.post(f"{address}/{endpoint}", json=data)
        except:             self.close_host_process(address, "Server did not respond", with_assert="Server did not respond.\nCheck console for details.")

        # clean up after request
        torch.cuda.memory.empty_cache()
        gc.collect()

        return result


    def launch_host(self, config):
        closed = False

        port = str(config.get("port"))
        backend = config.get("backend")
        current_config = self.configs.get(str(port))
        if current_config is None:
            closed = True
        else:
            process, backend, last_config, is_closed = current_config.get("process"), current_config.get("backend"), current_config.get("last_config"), current_config.get("closed")

            if is_closed:
                closed = True

            if not closed and (process is None or backend is None or last_config is None):
                self.close_host_process(port, "Invalid host configuration")
                closed = True

            if not closed and (backend != backend or len(backend) == 0):
                self.close_host_process(port, "Backend changed")
                closed = True

            if not closed:
                for k, v in last_config.items():
                    if k not in config:
                        self.close_host_process(port, "Updated configuration")
                        closed = True
                        break
                    if str(v) != str(config[k]):
                        self.close_host_process(port, "Updated configuration")
                        closed = True
                        break
                for k, v in config.items():
                    if k not in last_config:
                        self.close_host_process(port, "Updated configuration")
                        closed = True
                        break

        if backend in ["balanced"]:
            cmd = ["python3", f'{get_node_dir()}/multigpu_diffusion/host_{backend}.py', f'--port={port}']
        else:
            cmd = ["torchrun", f'--nproc_per_node={config["nproc_per_node"]}', f'--master-port={config["master_port"]}', f'{get_node_dir()}/multigpu_diffusion/host_{backend}.py', f'--port={port}']

        if len(config["cuda_visible_devices"]) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ.pop("CUDA_VISIBLE_DEVICES")

        print(f'Starting host:{str(cmd)}')
        process = subprocess.Popen(cmd)
        new_config = {
            "process": process,
            "backend": backend,
            "last_config": config,
            "closed": False,
            "pipeline": None,
        }
        self.__set_config_for_address(port, new_config)

        current = 0
        while True:
            try:
                response = self.get_from_address(LOCAL_HOST + port, "initialize")
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
            current += 1
            if current >= 30:
                self.close_host_process(port, "Timed out", with_assert="Failed to launch host within 30 seconds.\nCheck console for details.")

        # clean up after host launch
        torch.cuda.memory.empty_cache()
        gc.collect()

        return f"{LOCAL_HOST}{port}"


    def close_host_process(self, port, reason, wait_for_close=True, with_assert=None):
        if LOCAL_HOST in port: port = port.replace(LOCAL_HOST, "")

        try: self.get_from_address(LOCAL_HOST+port, "close")
        except: pass

        config = self.configs.get(port)
        process = config.get("process")
        if wait_for_close:  print(f'Synchronously stopping host process ({reason})')
        else:               print(f'Stopping host process ({reason})')
        def close(port, process):
            host = psutil.Process(process.pid)
            workers = [host] + host.children(recursive=True)
            while True:
                for w in workers:
                    result = subprocess.run(["kill", "-9", str(w.pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    time.sleep(3)
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.bind(("localhost", int(port)))
                    time.sleep(1)
                    s.close()
                    if self.configs.get(port) is not None: self.configs[port]["closed"] = True
                    print('Host has been stopped')
                    return
                except socket.error as e:
                    if e.errno == errno.EADDRINUSE:
                        print('Host still active - waiting for exit')
                        time.sleep(3)
                except Exception as ex:
                    print(f'Error occurred - waiting for exit\n{str(ex)}')
                    time.sleep(3)
        if not wait_for_close:      threading.Thread(target=close, args=(port, process,)).start()
        else:                       close(port, process)

        # clean up after closing host
        torch.cuda.memory.empty_cache()
        gc.collect()

        if with_assert is not None: assert False, with_assert
        return


    def get_result(self, address, data):
        config = self.__get_config_by_address(address)
        process = config["process"]
        last_config = config["last_config"]
        port = address.replace(LOCAL_HOST, "")

        try:
            response = self.post_to_address(address, "generate", data)
        except requests.exceptions.ConnectionError as e0:
            self.close_host_process(port, "Connection error", with_assert="Connection error.\nCheck console for details.")
        except Exception as e1:
            self.close_host_process(port, "Exception occurred while connecting", with_assert="Exception occurred while connecting.\nCheck console for details.")

        try:
            response_data = response.json()
            output_image_b64 = response_data.get("output")
            output_latent_b64 = response_data.get("latent")
            if output_image_b64 is None and output_latent_b64 is None:
                assert response_data is not None, "No response from host.\nCheck console for details."
                assert False, response_data.get("message")
            else:
                if output_latent_b64 is not None:   return output_image_b64, output_latent_b64
                else:                               return output_image_b64
        except Exception as e:
            run = False
            self.close_host_process(port, "Unknown error", with_assert=f"An error occurred while generating image.\nCheck console for details.\n\n{str(e)}")

