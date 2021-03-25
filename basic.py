from collections import namedtuple

import random
import gym
import vizdoomgym


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    """Store replay memory for DQN."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly sample previous transitions from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def trivial():
    # make deathmatch env
    env = gym.make("VizdoomDeathmatch-v0")
    env.reset()

    print("Available actions: {}".format(env.action_space))

    # for 1000 timesteps
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()

        # observation is screen info we want
        observation, reward, done, info = env.step(action)

        if done:
            observation = env.reset()

    env.close()


def main():
    """
    Running the program.
    """
    trivial()


if __name__ == "__main__":
    main()
