# CALVIN Environment 10

A Python 3.10+ compatible version of the CALVIN Environment for VR Data Collection and Rendering in robotics research.

## Overview

CALVIN Environment 10 is a modernized version of the original CALVIN environment, updated for Python 3.10+ compatibility and improved maintainability. It provides a simulation environment for robotics research with support for:

- Robot manipulation tasks
- VR data collection
- Camera rendering (static, gripper, tactile sensors)
- Scene management with objects, doors, buttons, switches, and lights
- PyBullet physics simulation

## Features

- **Python 3.10+ Compatible**: Uses modern Python features and type annotations
- **No OmegaConf Dependency**: Uses standard YAML instead of OmegaConf
- **Direct Instantiation**: Replaces Hydra instantiation with direct constructor calls
- **Improved Performance**: Faster object creation and configuration loading
- **Better Error Handling**: Comprehensive error messages and validation
- **Command Line Interface**: Easy-to-use CLI with argparse

## Installation

### Prerequisites

- Python 3.10 or higher
- PyBullet
- OpenGL/EGL support (for rendering)

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd calvin_env_10

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Install Dependencies Only

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from calvin_env_10 import get_env, PlayTableSimEnv

# Load environment from dataset
env = get_env("/path/to/dataset")

# Reset environment
obs = env.reset()

# Take action
action = [0.1, 0.0, 0.0, 0.0, 0.0, 1.0]  # [x, y, z, rx, ry, rz, gripper]
obs, reward, done, info = env.step(action)
```

### Command Line Interface

```bash
# Run with dataset path
python -m calvin_env_10.envs.play_table_env /path/to/dataset

# Run without GUI
python -m calvin_env_10.envs.play_table_env /path/to/dataset --no_gui

# Get help
python -m calvin_env_10.envs.play_table_env --help
```

## Project Structure

```
calvin_env_10/
├── __init__.py              # Package initialization
├── pyproject.toml           # Project configuration
├── requirements.txt         # Dependencies
├── README.md               # This file
├── envs/                   # Environment implementations
│   └── play_table_env.py   # Main environment class
├── robot/                  # Robot implementations
│   └── robot.py           # Robot base class
├── scene/                  # Scene management
│   └── play_table_scene.py # Scene implementation
├── camera/                 # Camera implementations
│   ├── camera.py          # Camera base class
│   ├── static_camera.py   # Static camera
│   ├── gripper_camera.py  # Gripper-mounted camera
│   └── tactile_sensor.py  # Tactile sensor
├── utils/                  # Utility functions
│   └── utils.py           # Common utilities
└── tests/                  # Test suite
    └── test_*.py          # Test files
```

## Configuration

The environment uses YAML configuration files. Example configuration:

```yaml
# Environment configuration
env:
  _target_: calvin_env_10.envs.play_table_env.PlayTableSimEnv
  bullet_time_step: 240.0
  use_vr: false
  show_gui: true
  use_scene_info: true
  use_egl: true
  control_freq: 30

# Robot configuration
robot:
  _target_: calvin_env_10.robot.robot.Robot
  filename: franka_panda/panda.urdf
  base_position: [0, 0, 0]
  base_orientation: [0, 0, 0]
  initial_joint_positions: [0, 0, 0, 0, 0, 0, 0]
  max_joint_force: 200.0
  gripper_force: 200
  arm_joint_ids: [0, 1, 2, 3, 4, 5, 6]
  gripper_joint_ids: [9, 10]
  gripper_joint_limits: [0, 0.04]
  tcp_link_id: 13
  end_effector_link_id: 7
  use_nullspace: true
  max_velocity: 2
  use_ik_fast: false
  euler_obs: true
```

## Development

### Setting up Development Environment

```bash
# Clone and install in development mode
git clone <repository-url>
cd calvin_env_10
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=calvin_env_10

# Run specific test file
pytest tests/test_environment.py
```

### Code Formatting

```bash
# Format code with black
black calvin_env_10/

# Sort imports with isort
isort calvin_env_10/

# Check code style with flake8
flake8 calvin_env_10/
```

## API Reference

### Main Classes

- `PlayTableSimEnv`: Main environment class implementing gym.Env
- `Robot`: Robot manipulation interface
- `PlayTableScene`: Scene management with objects and interactions
- `StaticCamera`: Static camera for environment observation
- `GripperCamera`: Camera mounted on robot gripper
- `TactileSensor`: Tactile sensor for contact detection

### Key Functions

- `get_env(dataset_path, obs_space=None, show_gui=True)`: Load environment from dataset
- `run_env(dataset_path, show_gui=True)`: Run environment with CLI interface
- `instantiate_from_config(config, **kwargs)`: Factory function for object creation

## Migration from Original CALVIN

This version is designed to be a drop-in replacement for the original CALVIN environment with the following changes:

1. **Import Changes**: Update imports to use `calvin_env_10` instead of `calvin_env`
2. **Configuration**: Uses standard YAML instead of OmegaConf
3. **Instantiation**: Direct constructor calls instead of Hydra instantiation
4. **Python Version**: Requires Python 3.10+ instead of Python 3.8

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the GPLv3 License - see the LICENSE file for details.

## Citation

If you use this environment in your research, please cite the original CALVIN paper:

```bibtex
@article{mees2022calvin,
  title={CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks},
  author={Mees, Oier and Hermann, Lukas and Rosete-Beas, Manuel and Burgard, Wolfram},
  journal={IEEE Robotics and Automation Letters},
  year={2022}
}
```

## Acknowledgments

- Original CALVIN Environment by Oier Mees, Lukas Hermann, and Wolfram Burgard
- PyBullet physics engine
- OpenAI Gym for the environment interface 