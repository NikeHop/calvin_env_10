"""
Tests for CALVIN Environment 10.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from calvin_env_10 import PlayTableSimEnv, get_env, instantiate_from_config


class TestPlayTableSimEnv:
    """Test cases for PlayTableSimEnv class."""

    def test_environment_creation(self):
        """Test that environment can be created with valid config."""
        # Mock configuration
        robot_cfg = {
            "_target_": "calvin_env_10.robot.robot.Robot",
            "filename": "test_robot.urdf",
            "base_position": [0, 0, 0],
            "base_orientation": [0, 0, 0],
            "initial_joint_positions": [0, 0, 0, 0, 0, 0, 0],
            "max_joint_force": 200.0,
            "gripper_force": 200,
            "arm_joint_ids": [0, 1, 2, 3, 4, 5, 6],
            "gripper_joint_ids": [9, 10],
            "gripper_joint_limits": [0, 0.04],
            "tcp_link_id": 13,
            "end_effector_link_id": 7,
            "use_nullspace": True,
            "max_velocity": 2,
            "use_ik_fast": False,
            "euler_obs": True,
        }

        scene_cfg = {
            "_target_": "calvin_env_10.scene.play_table_scene.PlayTableScene",
            "objects": {},
            "data_path": "test_data",
            "euler_obs": True,
            "global_scaling": 1.0,
            "surfaces": [],
        }

        cameras = {}

        # Mock PyBullet
        with patch("calvin_env_10.envs.play_table_env.p") as mock_p:
            mock_p.connect.return_value = 0
            mock_p.DIRECT = 0
            mock_p.GUI = 1
            mock_p.SHARED_MEMORY = 2

            # Mock other dependencies
            with patch(
                "calvin_env_10.envs.play_table_env.hydra.utils.instantiate"
            ) as mock_instantiate:
                mock_instantiate.side_effect = lambda cfg, **kwargs: Mock()

                env = PlayTableSimEnv(
                    robot_cfg=robot_cfg,
                    seed=42,
                    use_vr=False,
                    bullet_time_step=240.0,
                    cameras=cameras,
                    show_gui=False,
                    scene_cfg=scene_cfg,
                    use_scene_info=True,
                    use_egl=False,
                    control_freq=30,
                )

                assert env is not None
                assert env.cid == 0


class TestFactoryFunction:
    """Test cases for the instantiate_from_config factory function."""

    def test_robot_instantiation(self):
        """Test robot instantiation from config."""
        robot_config = {
            "_target_": "calvin_env_10.robot.robot.Robot",
            "filename": "test_robot.urdf",
            "base_position": [0, 0, 0],
            "base_orientation": [0, 0, 0],
            "initial_joint_positions": [0, 0, 0, 0, 0, 0, 0],
            "max_joint_force": 200.0,
            "gripper_force": 200,
            "arm_joint_ids": [0, 1, 2, 3, 4, 5, 6],
            "gripper_joint_ids": [9, 10],
            "gripper_joint_limits": [0, 0.04],
            "tcp_link_id": 13,
            "end_effector_link_id": 7,
            "use_nullspace": True,
            "max_velocity": 2,
            "use_ik_fast": False,
            "euler_obs": True,
            "cid": -1,
        }

        with patch("calvin_env_10.robot.robot.Robot") as mock_robot_class:
            mock_robot = Mock()
            mock_robot_class.return_value = mock_robot

            robot = instantiate_from_config(robot_config, cid=0)

            mock_robot_class.assert_called_once()
            assert robot == mock_robot

    def test_unknown_target_class(self):
        """Test that unknown target classes raise ValueError."""
        config = {"_target_": "unknown.module.Class", "param": "value"}

        with pytest.raises(ValueError, match="Unknown target class"):
            instantiate_from_config(config)

    def test_no_target_class(self):
        """Test that configs without _target_ are returned as-is."""
        config = {"param": "value"}

        result = instantiate_from_config(config)
        assert result == config


class TestEnvironmentInterface:
    """Test cases for environment interface compliance."""

    def test_gym_interface(self):
        """Test that environment implements gym.Env interface."""
        # This test ensures the environment follows the gym interface
        # The actual implementation would require more complex mocking
        assert hasattr(PlayTableSimEnv, "reset")
        assert hasattr(PlayTableSimEnv, "step")
        assert hasattr(PlayTableSimEnv, "render")
        assert hasattr(PlayTableSimEnv, "close")
        assert hasattr(PlayTableSimEnv, "seed")


if __name__ == "__main__":
    pytest.main([__file__])
