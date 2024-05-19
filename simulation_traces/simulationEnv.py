from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

from requestGenerator import getRequests

DEFAULT_X = np.pi
DEFAULT_Y = 1.0




def _step(tensordict):
    
    tensordict["time"] += 1

    N = tensordict["params","N"]
    M = tensordict["params","M"]
    time = tensordict["time"]

    action = tensordict["action"].squeeze(-1)
    
    requests = getRequests(time)

    while sum(requests)>0:
        request = np.choice(range(M), p=requests/np.sum(requests))

        pm_id = 0
        if np.sum(action[request]) == 0:
            pm_id = np.random.choice(range(M))
        else:
            pm_id = np.random.choice(range(M), p=action)
        self.PMs[pm_id].addRequest(self.apps[request])

        request = self.traceGenerator.getRequest()
    
    energy_cost = 0
    for pm in self.PMs:
        energy_cost += pm.powerUsage()
    
    QoS = 0
    # TODO get QoS
    
    # apply penalty for QoS if CPU load is higher than CPU capacity
    QoS_penalty = 0
    for pm in self.PMs:
        if pm.CPU_load > pm.CPU*T:
            QoS_penalty += pm.CPU_load - pm.CPU*T

    reward = -energy_cost - LAMBDA*QoS - np.exp(LAMBDA_2*QoS_penalty)
    
    
    
    u = u.clamp(-tensordict["params", "max_torque"], tensordict["params", "max_torque"])
    costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

    new_thdot = (
        thdot
        + (3 * g_force / (2 * length) * th.sin() + 3.0 / (mass * length**2) * u) * dt
    )
    new_thdot = new_thdot.clamp(
        -tensordict["params", "max_speed"], tensordict["params", "max_speed"]
    )
    new_th = th + new_thdot * dt
    reward = -costs.view(*tensordict.shape, 1)
    done = torch.zeros_like(reward, dtype=torch.bool)
    out = TensorDict(
        {
            "th": new_th,
            "thdot": new_thdot,
            "params": tensordict["params"],
            "reward": reward,
            "done": done,
        },
        tensordict.shape,
    )
    return out



def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty():
        # if no ``tensordict`` is passed, we generate a single set of hyperparameters
        # Otherwise, we assume that the input ``tensordict`` contains all the relevant
        # parameters to get started.
        tensordict = self.gen_params(batch_size=self.batch_size)

    self.simulation = Simulation()

    out = TensorDict(
        {
            "data": Data()
        },
        batch_size=tensordict.shape,
    )
    return out


def _make_spec(self, td_params):
    # Under the hood, this will populate self.output_spec["observation"]
    self.observation_spec = CompositeSpec(
        th=BoundedTensorSpec(
            low=-torch.pi,
            high=torch.pi,
            shape=(),
            dtype=torch.float32,
        ),
        thdot=BoundedTensorSpec(
            low=-td_params["params", "max_speed"],
            high=td_params["params", "max_speed"],
            shape=(),
            dtype=torch.float32,
        ),
        # we need to add the ``params`` to the observation specs, as we want
        # to pass it at each step during a rollout
        params=make_composite_from_td(td_params["params"]),
        shape=(),
    )
    # since the environment is stateless, we expect the previous output as input.
    # For this, ``EnvBase`` expects some state_spec to be available
    self.state_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec when
    # `self.action_spec = spec` will be called supported
    self.action_spec = BoundedTensorSpec(
        low=-td_params["params", "max_torque"],
        high=td_params["params", "max_torque"],
        shape=(1,),
        dtype=torch.float32,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))


def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite

def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    self.rng = rng


def gen_params(batch_size=None) -> TensorDictBase:
    """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "max_speed": 8,
                    "max_torque": 2.0,
                    "dt": 0.05,
                    "g": g,
                    "m": 1.0,
                    "l": 1.0,
                },
                [],
            )
        },
        [],
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td


class SimulationEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helpers: _make_step and gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed







class VmasWrapper(_EnvWrapper):
    """Vmas environment wrapper.

    GitHub: https://github.com/proroklab/VectorizedMultiAgentSimulator

    Paper: https://arxiv.org/abs/2207.03530

    Args:
        env (``vmas.simulator.environment.environment.Environment``): the vmas environment to wrap.

    Keyword Args:
        num_envs (int): Number of vectorized simulation environments. VMAS perfroms vectorized simulations using PyTorch.
            This argument indicates the number of vectorized environments that should be simulated in a batch. It will also
            determine the batch size of the environment.
        device (torch.device, optional): Device for simulation. Defaults to the default device. All the tensors created by VMAS
            will be placed on this device.
        continuous_actions (bool, optional): Whether to use continuous actions. Defaults to ``True``. If ``False``, actions
            will be discrete. The number of actions and their size will depend on the chosen scenario.
            See the VMAS repository for more info.
        max_steps (int, optional): Horizon of the task. Defaults to ``None`` (infinite horizon). Each VMAS scenario can
            be terminating or not. If ``max_steps`` is specified,
            the scenario is also terminated (and the ``"terminated"`` flag is set) whenever this horizon is reached.
            Unlike gym's ``TimeLimit`` transform or torchrl's :class:`~torchrl.envs.transforms.StepCounter`,
            this argument will not set the ``"truncated"`` entry in the tensordict.
        categorical_actions (bool, optional): if the environment actions are discrete, whether to transform
            them to categorical or one-hot. Defaults to ``True``.
        group_map (MarlGroupMapType or Dict[str, List[str]], optional): how to group agents in tensordicts for
            input/output. By default, if the agent names follow the ``"<name>_<int>"``
            convention, they will be grouped by ``"<name>"``. If they do not follow this convention, they will be all put
            in one group named ``"agents"``.
            Otherwise, a group map can be specified or selected from some premade options.
            See :class:`~torchrl.envs.utils.MarlGroupMapType` for more info.

    Attributes:
        group_map (Dict[str, List[str]]): how to group agents in tensordicts for
            input/output. See :class:`~torchrl.envs.utils.MarlGroupMapType` for more info.
        agent_names (list of str): names of the agent in the environment
        agent_names_to_indices_map (Dict[str, int]): dictionary mapping agent names to their index in the environment
        unbatched_action_spec (TensorSpec): version of the spec without the vectorized dimension
        unbatched_observation_spec (TensorSpec): version of the spec without the vectorized dimension
        unbatched_reward_spec (TensorSpec): version of the spec without the vectorized dimension
        het_specs (bool): whether the enviornment has any lazy spec
        het_specs_map (Dict[str, bool]): dictionary mapping each group to a flag representing of the group has lazy specs
        available_envs (List[str]): the list of the scenarios available to build.

    .. warning::
        VMAS returns a single ``done`` flag which does not distinguish between
        when the env reached ``max_steps`` and termination.
        If you deem the ``truncation`` signal necessary, set ``max_steps`` to
        ``None`` and use a :class:`~torchrl.envs.transforms.StepCounter` transform.

    Examples:
        >>>  env = VmasWrapper(
        ...      vmas.make_env(
        ...          scenario="flocking",
        ...          num_envs=32,
        ...          continuous_actions=True,
        ...          max_steps=200,
        ...          device="cpu",
        ...          seed=None,
        ...          # Scenario kwargs
        ...          n_agents=5,
        ...      )
        ...  )
        >>>  print(env.rollout(10))
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        action: Tensor(shape=torch.Size([32, 10, 5, 2]), device=cpu, dtype=torch.float32, is_shared=False),
                        info: TensorDict(
                            fields={
                                agent_collision_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                                agent_distance_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([32, 10, 5]),
                            device=cpu,
                            is_shared=False),
                        observation: Tensor(shape=torch.Size([32, 10, 5, 18]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([32, 10, 5]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        agents: TensorDict(
                            fields={
                                info: TensorDict(
                                    fields={
                                        agent_collision_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                                        agent_distance_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                                    batch_size=torch.Size([32, 10, 5]),
                                    device=cpu,
                                    is_shared=False),
                                observation: Tensor(shape=torch.Size([32, 10, 5, 18]), device=cpu, dtype=torch.float32, is_shared=False),
                                reward: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([32, 10, 5]),
                            device=cpu,
                            is_shared=False),
                        done: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        terminated: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([32, 10]),
                    device=cpu,
                    is_shared=False),
                terminated: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([32, 10]),
            device=cpu,
            is_shared=False)
    """

    git_url = "https://github.com/proroklab/VectorizedMultiAgentSimulator"
    libname = "vmas"

    @property
    def lib(self):
        import vmas

        return vmas

    @_classproperty
    def available_envs(cls):
        if not _has_vmas:
            return []
        return list(_get_envs())

    def __init__(
        self,
        env: "vmas.simulator.environment.environment.Environment" = None,  # noqa
        categorical_actions: bool = True,
        group_map: MarlGroupMapType | Dict[str, List[str]] | None = None,
        **kwargs,
    ):
        if env is not None:
            kwargs["env"] = env
            if "device" in kwargs.keys() and kwargs["device"] != str(env.device):
                raise TypeError("Env device is different from vmas device")
            kwargs["device"] = str(env.device)
        self.group_map = group_map
        self.categorical_actions = categorical_actions
        super().__init__(**kwargs, allow_done_after_reset=True)

    def _build_env(
        self,
        env: "vmas.simulator.environment.environment.Environment",  # noqa
        from_pixels: bool = False,
        pixels_only: bool = False,
    ):
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only

        # TODO pixels
        if self.from_pixels:
            raise NotImplementedError("vmas rendering not yet implemented")

        # Adjust batch size
        if len(self.batch_size) == 0:
            # Batch size not set
            self.batch_size = torch.Size((env.num_envs,))
        elif len(self.batch_size) == 1:
            # Batch size is set
            if not self.batch_size[0] == env.num_envs:
                raise TypeError(
                    "Batch size used in constructor does not match vmas batch size."
                )
        else:
            raise TypeError(
                "Batch size used in constructor is not compatible with vmas."
            )

        return env

    def _get_default_group_map(self, agent_names: List[str]):
        # This function performs the default grouping in vmas.
        # Agents with names "<name>_<int>" will be grouped in group name "<name>".
        # If any of the agents does not follow the naming convention, we fall back
        # back on having all agents in one group named "agents".
        group_map = {}
        follows_convention = True
        for agent_name in agent_names:
            # See if the agent follows the convention "<name>_<int>"
            agent_name_split = agent_name.split("_")
            if len(agent_name_split) == 1:
                follows_convention = False
            follows_convention = follows_convention and agent_name_split[-1].isdigit()

            if not follows_convention:
                break

            # Group it with other agents that follow the same convention
            group_name = "_".join(agent_name_split[:-1])
            if group_name in group_map:
                group_map[group_name].append(agent_name)
            else:
                group_map[group_name] = [agent_name]

        if not follows_convention:
            group_map = MarlGroupMapType.ALL_IN_ONE_GROUP.get_group_map(agent_names)

        # For BC-compatibility rename the "agent" group to "agents"
        if "agent" in group_map and len(group_map) == 1:
            agent_group = group_map["agent"]
            group_map["agents"] = agent_group
            del group_map["agent"]
        return group_map

    def _make_specs(
        self, env: "vmas.simulator.environment.environment.Environment"  # noqa
    ) -> None:
        # Create and check group map
        self.agent_names = [agent.name for agent in self.agents]
        self.agent_names_to_indices_map = {
            agent.name: i for i, agent in enumerate(self.agents)
        }
        if self.group_map is None:
            self.group_map = self._get_default_group_map(self.agent_names)
        elif isinstance(self.group_map, MarlGroupMapType):
            self.group_map = self.group_map.get_group_map(self.agent_names)
        check_marl_grouping(self.group_map, self.agent_names)

        self.unbatched_action_spec = CompositeSpec(device=self.device)
        self.unbatched_observation_spec = CompositeSpec(device=self.device)
        self.unbatched_reward_spec = CompositeSpec(device=self.device)

        self.het_specs = False
        self.het_specs_map = {}
        for group in self.group_map.keys():
            (
                group_observation_spec,
                group_action_spec,
                group_reward_spec,
                group_info_spec,
            ) = self._make_unbatched_group_specs(group)
            self.unbatched_action_spec[group] = group_action_spec
            self.unbatched_observation_spec[group] = group_observation_spec
            self.unbatched_reward_spec[group] = group_reward_spec
            if group_info_spec is not None:
                self.unbatched_observation_spec[(group, "info")] = group_info_spec
            group_het_specs = isinstance(
                group_observation_spec, LazyStackedCompositeSpec
            ) or isinstance(group_action_spec, LazyStackedCompositeSpec)
            self.het_specs_map[group] = group_het_specs
            self.het_specs = self.het_specs or group_het_specs

        self.unbatched_done_spec = CompositeSpec(
            {
                "done": DiscreteTensorSpec(
                    n=2,
                    shape=torch.Size((1,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
        )

        self.action_spec = self.unbatched_action_spec.expand(
            *self.batch_size, *self.unbatched_action_spec.shape
        )
        self.observation_spec = self.unbatched_observation_spec.expand(
            *self.batch_size, *self.unbatched_observation_spec.shape
        )
        self.reward_spec = self.unbatched_reward_spec.expand(
            *self.batch_size, *self.unbatched_reward_spec.shape
        )
        self.done_spec = self.unbatched_done_spec.expand(
            *self.batch_size, *self.unbatched_done_spec.shape
        )

    def _make_unbatched_group_specs(self, group: str):
        # Agent specs
        action_specs = []
        observation_specs = []
        reward_specs = []
        info_specs = []
        for agent_name in self.group_map[group]:
            agent_index = self.agent_names_to_indices_map[agent_name]
            agent = self.agents[agent_index]
            action_specs.append(
                CompositeSpec(
                    {
                        "action": _vmas_to_torchrl_spec_transform(
                            self.action_space[agent_index],
                            categorical_action_encoding=self.categorical_actions,
                            device=self.device,
                        )  # shape = (n_actions_per_agent,)
                    },
                )
            )
            observation_specs.append(
                CompositeSpec(
                    {
                        "observation": _vmas_to_torchrl_spec_transform(
                            self.observation_space[agent_index],
                            device=self.device,
                            categorical_action_encoding=self.categorical_actions,
                        )  # shape = (n_obs_per_agent,)
                    },
                )
            )
            reward_specs.append(
                CompositeSpec(
                    {
                        "reward": UnboundedContinuousTensorSpec(
                            shape=torch.Size((1,)),
                            device=self.device,
                        )  # shape = (1,)
                    }
                )
            )
            agent_info = self.scenario.info(agent)
            if len(agent_info):
                info_specs.append(
                    CompositeSpec(
                        {
                            key: UnboundedContinuousTensorSpec(
                                shape=_selective_unsqueeze(
                                    value, batch_size=self.batch_size
                                ).shape[1:],
                                device=self.device,
                                dtype=torch.float32,
                            )
                            for key, value in agent_info.items()
                        },
                    ).to(self.device)
                )

        # Create multi-agent specs
        group_action_spec = torch.stack(
            action_specs, dim=0
        )  # shape = (n_agents, n_actions_per_agent)
        group_observation_spec = torch.stack(
            observation_specs, dim=0
        )  # shape = (n_agents, n_obs_per_agent)
        group_reward_spec = torch.stack(reward_specs, dim=0)  # shape = (n_agents, 1)
        group_info_spec = None
        if len(info_specs):
            group_info_spec = torch.stack(info_specs, dim=0)

        return (
            group_observation_spec,
            group_action_spec,
            group_reward_spec,
            group_info_spec,
        )

    def _check_kwargs(self, kwargs: Dict):
        vmas = self.lib

        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        env = kwargs["env"]
        if not isinstance(env, vmas.simulator.environment.Environment):
            raise TypeError(
                "env is not of type 'vmas.simulator.environment.Environment'."
            )

    def _init_env(self) -> Optional[int]:
        pass

    def _set_seed(self, seed: Optional[int]):
        self._env.seed(seed)

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:
        if tensordict is not None and "_reset" in tensordict.keys():
            _reset = tensordict.get("_reset")
            envs_to_reset = _reset.squeeze(-1)
            if envs_to_reset.all():
                self._env.reset(return_observations=False)
            else:
                for env_index, to_reset in enumerate(envs_to_reset):
                    if to_reset:
                        self._env.reset_at(env_index, return_observations=False)
        else:
            self._env.reset(return_observations=False)

        obs, dones, infos = self._env.get_from_scenario(
            get_observations=True,
            get_infos=True,
            get_rewards=False,
            get_dones=True,
        )
        dones = self.read_done(dones)

        source = {"done": dones, "terminated": dones.clone()}
        for group, agent_names in self.group_map.items():
            agent_tds = []
            for agent_name in agent_names:
                i = self.agent_names_to_indices_map[agent_name]

                agent_obs = self.read_obs(obs[i])
                agent_info = self.read_info(infos[i])
                agent_td = TensorDict(
                    source={
                        "observation": agent_obs,
                    },
                    batch_size=self.batch_size,
                    device=self.device,
                )
                if agent_info is not None:
                    agent_td.set("info", agent_info)
                agent_tds.append(agent_td)

            agent_tds = LazyStackedTensorDict.maybe_dense_stack(agent_tds, dim=1)
            if not self.het_specs_map[group]:
                agent_tds = agent_tds.to_tensordict()
            source.update({group: agent_tds})

        tensordict_out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        agent_indices = {}
        action_list = []
        n_agents = 0
        for group, agent_names in self.group_map.items():
            group_action = tensordict.get((group, "action"))
            group_action_list = list(self.read_action(group_action, group=group))
            agent_indices.update(
                {
                    self.agent_names_to_indices_map[agent_name]: i + n_agents
                    for i, agent_name in enumerate(agent_names)
                }
            )
            n_agents += len(agent_names)
            action_list += group_action_list
        action = [action_list[agent_indices[i]] for i in range(self.n_agents)]

        obs, rews, dones, infos = self._env.step(action)

        dones = self.read_done(dones)

        source = {"done": dones, "terminated": dones.clone()}
        for group, agent_names in self.group_map.items():
            agent_tds = []
            for agent_name in agent_names:
                i = self.agent_names_to_indices_map[agent_name]

                agent_obs = self.read_obs(obs[i])
                agent_rew = self.read_reward(rews[i])
                agent_info = self.read_info(infos[i])

                agent_td = TensorDict(
                    source={
                        "observation": agent_obs,
                        "reward": agent_rew,
                    },
                    batch_size=self.batch_size,
                    device=self.device,
                )
                if agent_info is not None:
                    agent_td.set("info", agent_info)
                agent_tds.append(agent_td)

            agent_tds = LazyStackedTensorDict.maybe_dense_stack(agent_tds, dim=1)
            if not self.het_specs_map[group]:
                agent_tds = agent_tds.to_tensordict()
            source.update({group: agent_tds})

        tensordict_out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out

    def read_obs(
        self, observations: Union[Dict, torch.Tensor]
    ) -> Union[Dict, torch.Tensor]:
        if isinstance(observations, torch.Tensor):
            return _selective_unsqueeze(observations, batch_size=self.batch_size)
        return TensorDict(
            source={key: self.read_obs(value) for key, value in observations.items()},
            batch_size=self.batch_size,
        )

    def read_info(self, infos: Dict[str, torch.Tensor]) -> torch.Tensor:
        if len(infos) == 0:
            return None
        infos = TensorDict(
            source={
                key: _selective_unsqueeze(
                    value.to(torch.float32), batch_size=self.batch_size
                )
                for key, value in infos.items()
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        return infos

    def read_done(self, done):
        done = _selective_unsqueeze(done, batch_size=self.batch_size)
        return done

    def read_reward(self, rewards):
        rewards = _selective_unsqueeze(rewards, batch_size=self.batch_size)
        return rewards

    def read_action(self, action, group: str = "agents"):
        if not self.continuous_actions and not self.categorical_actions:
            action = self.unbatched_action_spec[group, "action"].to_categorical(action)
        agent_actions = action.unbind(dim=1)
        return agent_actions

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_envs={self.num_envs}, n_agents={self.n_agents},"
            f" batch_size={self.batch_size}, device={self.device})"
        )

    def to(self, device: DEVICE_TYPING) -> EnvBase:
        self._env.to(device)
        return super().to(device)