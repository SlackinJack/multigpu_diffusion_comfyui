# multigpu_diffusion_comfyui

ComfyUI nodes to run [multigpu_diffusion](https://github.com/SlackinJack/multigpu_diffusion).

(Basically, a ComfyUI front-end to Diffusers, with support for multi-GPU environments.)


![Screenshot 1](/.gallery/workflow_example.png?raw=true "Screenshot 1")


## Notes:
- Requires **Diffusers-format** checkpoints/models.
- Does not (and will never) use native sampling.
- Windows and macOS are not (and probably never will be) supported.
- **This node uses psutil/subprocess to manage the host scripts.** You can find the calls in `modules/host_manager.py`.


## Setup:
1. `cd` to `custom_nodes/multigpu_diffusion_comfyui`.
2. (Review and) run `setup.sh`.


## Debugging:
- If you run into issues, you can manually curl to the hosts (e.g., `curl localhost:{port}/close` to manually shut down the host).


## Known Issues:
- Running balanced host will create device errors, use another host for now.


## Test Environment:
- ComfyUI 0.12.3, front-end 1.16.9
- 4x Nvidia Tesla T4
- Python 3.14.4
- Torch 2.14.0
