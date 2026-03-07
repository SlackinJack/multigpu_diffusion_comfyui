import errno
import json
import logging
import os
import psutil
import requests
import socket
import subprocess
import threading
import time
import torch
import traceback


from ..multigpu_diffusion.modules.utils import *
from ..nodes.data_types import *


LOCAL_HOST = "http://localhost:"


class HostManager:
    def __init__(self):
        self.configs = {}
        #{
        #    "port": {
        #        "process": process,
        #        "backend": backend,
        #        "last_config": last_config,
        #        "pipeline": pipeline,
        #    }
        #}
        logger = logging.getLogger(name="manager")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt=f'[ Mngr ]: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        self.logger = logger
        return


    def log(self, text):
        self.logger.info(text)


    def __get_config_by_address(self, address):
        port = address.replace(LOCAL_HOST, "")
        return self.configs.get(port)


    def __set_config_for_address(self, address, config):
        if LOCAL_HOST in address: address = address.replace(LOCAL_HOST, "")
        self.configs[address] = config
        return


    def __update_config_pipeline(self, address, value):
        # config = self.__get_config_by_address(address)
        address = address.replace(LOCAL_HOST, "")
        config = self.configs.get(address)
        if config is None:  self.close_host_process(address, "Host not active", with_assert="Host not active.\nCheck console for details.")
        else:               config["pipeline"] = value
        return


    def get_from_address(self, address, endpoint, allow_error=False, pbar=None):
        if not LOCAL_HOST in address: address = LOCAL_HOST + address
        if endpoint not in ["initialize", "progress"]:
            self.log(f"ℹ️ Sending GET request to: {address}/{endpoint}")

        results = [None]
        def get():
            nonlocal self, address, endpoint, allow_error, results
            try:
                results[0] = requests.get(f"{address}/{endpoint}")
            except:
                if allow_error == False:
                    self.log(f"{traceback.format_exc()}")
                    self.close_host_process(address, "Server did not respond")
            return

        # result = requests.get(f"{address}/{endpoint}")
        thread = threading.Thread(target=get,)
        thread.start()
        if pbar is not None:
            while thread.is_alive():
                try:
                    response = requests.get(f"{address}/progress")
                    if response.status_code == 200:
                        pbar.update_absolute(int(response.text))
                except:
                    pass
                time.sleep(1)
        thread.join()
        return results[0]


    def post_to_address(self, address, endpoint, data, pbar=None):
        if not LOCAL_HOST in address: address = LOCAL_HOST + address
        match endpoint:
            case "apply":   self.__update_config_pipeline(address, data)
            case _:         pass
        self.log(f"ℹ️ Sending POST request to: {address}/{endpoint}")

        results = [None]
        def post():
            nonlocal self, address, endpoint, data, results
            try:
                results[0] = requests.post(f"{address}/{endpoint}", json=data)
            except:
                self.close_host_process(address, "Server did not respond", with_assert="Server did not respond.\nCheck console for details.")
                return None

        # result = requests.post(f"{address}/{endpoint}", json=data)
        thread = threading.Thread(target=post,)
        thread.start()
        if pbar is not None:
            while thread.is_alive():
                try:
                    response = requests.get(f"{address}/progress")
                    if response.status_code == 200:
                        pbar.update_absolute(int(response.text))
                except:
                    pass
                time.sleep(1)
        thread.join()
        return results[0]


    def launch_host(self, config, pbar=None):
        port = str(config.get("port"))
        backend = config.get("backend")
        current_config = self.configs.get(str(port))
        if current_config is not None:
            closed = False
            process, backend, last_config = current_config.get("process"), current_config.get("backend"), current_config.get("last_config")

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

        os.environ["OMP_NUM_THREADS"] = "1"
        if len(config["cuda_visible_devices"]) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ.pop("CUDA_VISIBLE_DEVICES")

        # NOTE: comment-out for debug
        os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
        os.environ["DIFFUSERS_VERBOSITY"] = "critical"
        os.environ["DIFFUSERS_NO_ADVISORY_WARNINGS"] = "1"
        os.environ["PYTHONWARNINGS"] = "ignore"

        # TODO: implement later, currently consistently slower than torchrun
        # has_accelerate = False
        # try:
        #     import accelerate
        #     has_accelerate = True
        # except: pass

        match backend:
            case "asyncdiff":
                # if has_accelerate == True:  cmd = ["accelerate", "launch",  f'--main_process_port={config["master_port"]}', f'--num_processes={config["nproc_per_node"]}']
                # else:                       cmd = ["torchrun",              f'--master-port={config["master_port"]}',       f'--nproc_per_node={config["nproc_per_node"]}']
                cmd = ["torchrun", f'--master-port={config["master_port"]}']
                if len(config["cuda_visible_devices"]) > 0:
                    cmd += [f'--nproc_per_node={len(config["cuda_visible_devices"].split(","))}']
                else:
                    cmd += [f'--nproc_per_node={torch.cuda.device_count()}']
            case _:
                cmd = ["python3"]
                # raise NotImplementedError
        cmd += [f'{get_node_dir()}/multigpu_diffusion/host_{backend}.py', f'--port={port}']

        cmd_string = ""
        for c in cmd: cmd_string += "\n        " + str(c)
        self.log(f'🟢 Starting host: {cmd_string}')

        process = subprocess.Popen(cmd)
        new_config = {
            "process": process,
            "backend": backend,
            "last_config": config,
            "pipeline": None,
        }
        self.__set_config_for_address(port, new_config)

        current = 0
        timeout = 30
        while True:
            try:
                response = self.get_from_address(LOCAL_HOST + port, "initialize", allow_error=True)
                if response is not None and response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
            if pbar is not None:
                pbar.update_absolute(int(current/timeout))
            current += 1
            if current >= timeout:
                self.close_host_process(port, "Timed out", with_assert="Failed to launch host within 30 seconds.\nCheck console for details.")

        return f"{LOCAL_HOST}{port}"


    def close_host_process(self, port, reason, wait_for_close=True, with_assert=None):
        if LOCAL_HOST in port: port = port.replace(LOCAL_HOST, "")

        self.get_from_address(LOCAL_HOST+port, "close", allow_error=True)

        config = self.configs.get(port)
        if config is None:
            self.log(f'❓ No host config - assuming host {port} is already closed')
        else:
            process = config.get("process")
            if wait_for_close:  self.log(f'🟡 Synchronously stopping host {port} process - {reason}')
            else:               self.log(f'🟡 Stopping host {port} process - {reason}')
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
                        if self.configs.get(port) is not None:
                            del self.configs[port]
                        self.log(f'🛑 Host {port} has been stopped')
                        break
                    except socket.error as e:
                        if e.errno == errno.EADDRINUSE:
                            self.log(f'⏳ Host {port} still active - waiting for exit')
                            time.sleep(3)
                    except Exception as ex:
                        self.log(f'❌ Error occurred - waiting for host {port} exit\n{str(ex)}')
                        time.sleep(3)

            result = subprocess.run(["curl", f'{LOCAL_HOST}{port}/close'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if not wait_for_close:      threading.Thread(target=close, args=(port, process,)).start()
            else:                       close(port, process)

        assert with_assert is None, with_assert
        return


    def get_result(self, address, data, pbar=None):
        port = address.replace(LOCAL_HOST, "")
        config = self.__get_config_by_address(address)
        if config is None: self.close_host_process(port, "Host not active", with_assert="Host not active.\nCheck console for details.")
        process = config["process"]
        last_config = config["last_config"]

        try:
            response = self.post_to_address(address, "generate", data, pbar=pbar)
        except requests.exceptions.ConnectionError as e0:
            self.close_host_process(port, "Connection error", with_assert="Connection error.\nCheck console for details.")
        except Exception as e1:
            self.close_host_process(port, "Exception occurred while connecting", with_assert="Exception occurred while connecting.\nCheck console for details.")

        try:
            response_data = response.json()
            assert response_data is not None, "No response from host.\nCheck console for details."
            output_image_b64 = response_data.get("output")
            output_latent_b64 = response_data.get("latent")
            assert output_image_b64 is not None or output_latent_b64 is not None, response_data.get("message")
            if output_latent_b64 is not None:   return output_image_b64, output_latent_b64
            else:                               return output_image_b64
        except Exception as e:
            run = False
            self.close_host_process(port, "Unknown error", with_assert=f"An error occurred while generating image.\nCheck console for details.\n\n{str(e)}")

