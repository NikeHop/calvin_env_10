# CALVIN Environment (Python 3.10)

The CALVIN environment with updated dependencies. For the original environment + dataset see [here](https://github.com/mees/calvin)

Some of the changes: 
* Removed [tacto](https://github.com/facebookresearch/tacto) dependency. No tactile info such as `rgb_tactile` and `depth_tactile` are available. 
* Updated the dependencies of [urdfpy](https://github.com/NikeHop/urdfpy.git).
* Simplified instantiation of the environment and multistep evaluation.

## Usage 

Install the environment as a dependency via uv or pip.

```sh
uv pip install https://github.com/NikeHop/calvin_env_10.git
```

Create an instance of the environment 

```python
from calvin_env_10.envs.play_table_env import get_env 

task = "task_D" # available tasks: task_A, task_B, task_C

env = get_env(task,show_gui=False)

info = env.reset()


```

Run multistep evaluation by wrapping your policy.

```python

 
```


## Citation

If you use this environment in your research, please cite the original CALVIN paper:

```bibtex
@article{mees2022calvin,
  title={Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks},
  author={Mees, Oier and Hermann, Lukas and Rosete-Beas, Erick and Burgard, Wolfram},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={3},
  pages={7327--7334},
  year={2022},
  publisher={IEEE}
}
```

