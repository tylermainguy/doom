from collections import namedtuple

import random
import gym
import vizdoomgym
from torchvision import transforms
from dqn import DQN


def train(env, predictor, target):
    """
    Train the DQN. Assuming single episode, for now.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Available actions: {}".format(env.action_space))

    # for 1000 timesteps
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()

        # observation is screen info we want
        observation, reward, done, info = env.step(action)

        image = transform(observation)

        # output = predictor(image)

        # normalize observation for network
        if done:
            observation = env.reset()

    env.close()


def main():
    """
    Running the program.
    """
    # make deathmatch env
    env = gym.make("VizdoomDeathmatch-v0")
    env.reset()

    num_actions = env.action_space.n

    predictor = DQN(240, 320, num_actions)
    target = DQN(240, 320, num_actions)

    # weights of target should be the same
    target.load_state_dict(predictor.state_dict())
    target.eval()

    train(env, predictor, target)


if __name__ == "__main__":
    main()
