from environment.scheduling_env import SchedulingEnv

from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    env = SchedulingEnv()
    parallel_api_test(env, num_cycles=1_000_000)
