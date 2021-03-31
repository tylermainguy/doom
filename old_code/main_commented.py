##Imports

import math
import random
from collections import deque, namedtuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import vizdoomgym
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from average_meter import AverageMeter
from dqn import DQN
from replay_buffer import ReplayBuffer

# Initialize tranisiton memory  tuple
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

## Global Parameter Initilization
"""
BATCH_SIZE : The number of samples to be used as input when training the model
GAMMA :  The  
"""
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


def generate_params():
    """
    Generate parameters to be used by model.
    """
    params = {}

    params["timesteps"] = 1000
    params["episodes"] = 500
    params["epochs"] = 5
    params["stack_size"] = 4
    params["skip_frames"] = 4
    params["batch_size"] = 64

    return params


def preprocess(observation):
    """
    Preprocess images to be used by DQN.
    This function will take a screen grab of the current time step of the game
    and transform it from a RGB image to greyscale, while resizing the height
    and width.
    """
    # convert to grayscale, and reshape to be quarter of original size
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Grayscale(), transforms.Resize((60, 80))]
    )

    # return the transformed image
    return transform(observation)


def update_stack(frame_stack, observation):
    """
    Update the frame stack with the an observation
    This function will create stacks of four consequtive
    images for the model to learn from.
    """
    image = preprocess(observation)

    frame_stack.append(image)


def select_action(state, timesteps, policy_net, num_actions):
    """
    Select_action uses a epsilon greedy method to select and action.
    This involves both greedy action selection and random action
    selection for exploration of our agent.
    """
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * timesteps / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(num_actions)]], dtype=torch.long)


def optimize_dqn(replay_buffer, target_net, pred_net, optim, params):
    """
    Perform optimization on the DQN given a batch of randomly
    sampled transitions from the replay buffer.

    Inputs:
        replay_buffer: buffer containing history of transitions
        params: dictionary containing parameters for network
        target_net: the network for used to compute the q-value of next states
        pred_net : The main deep Q network
        optim : the optimizer used to minimize the model loss

    This function contains the code the calculates the models Q-values for
    the current and next state. It also calcuates the subsequent huber loss.
    """

    # only want to update when we have enough transitions to sample
    if len(replay_buffer) < params["batch_size"] * 2:
        return

    # sample batch, and transpose batch
    sample = replay_buffer.sample(params["batch_size"])
    batch = Transition(*zip(*sample))

    # mask any transitions that have None (indicate transition to terminal state)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
    )
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

    # get batches of states, actions, and rewards for DQN
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # print("State size: {}".format(state_batch.shape))

    # Calculate the Q-value of the current state-action pair using the model.
    state_action_values = pred_net(state_batch).gather(1, action_batch)

    # calculate values for all possible next states. First check if next states
    # are non-final using the non_final_mask. The expected value for each action
    # is then caluclated via the target-net, and selected using a max function.
    next_state_values = torch.zeros(params["batch_size"])
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA).unsqueeze(1) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    print("Loss: {}".format(loss.item()))

    # Optimize the model
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pred_net.parameters(), 0.25)
    # for param in pred_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optim.step()


def train(params):
    """
    This function contains the main loop for the model training.
    First the replay buffer is initialized to store past transitions.
    It is done using a outer loop of the number of episodes, which is the
    number of times we want to run a game instance. The environment is
    reset after every episode.
    The inner loop consists of the time steps within the game, where a stack
    of 4 consequtive images are created to be used as input for the model.
    Every 100 time steps the main network is optimized using a random sample
    batch of image stacks from the replay buffer. The target_net, used to
    calculate expected values, is optimized every 1000 time steps

    Input
        params: the model paramteres to be used as defined by generate_params()
    """
    # Initialize the environment
    env = gym.make("VizdoomBasic-v0")
    num_actions = env.action_space.n

    # Intitialize both deep Q networks
    target_net = DQN(60, 80, num_actions=num_actions)
    pred_net = DQN(60, 80, num_actions=num_actions)

    # Create env, initialize the starting state
    target_net.load_state_dict(pred_net.state_dict())
    target_net.eval()

    # Initialize optimizer
    optim = torch.optim.Adam(pred_net.parameters())

    # Initialize replay memory
    replay_buffer = ReplayBuffer(10000)

    for episode in tqdm(range(params["episodes"]), desc="episodes", unit="episodes"):

        done = False
        env.reset()

        # Initialize frame stack
        frame_stack = deque(maxlen=params["stack_size"])

        # For frame skipping
        num_skipped = 0
        timestep = 0

        action = env.action_space.sample()

        while not done:
            env.render()
            # print(timestep)

            # execute action and observe reward
            # observation is screen info we want
            # print("ACTION: {}".format(action))
            observation, reward, done, _ = env.step(action)

            # if reward != 1.0:
            #     print(reward)
            #     print("action: {}".format(action))
            # only want to stack every four frames
            # if (timestep == 0) or (num_skipped == params["skip_frames"] - 1):
            # reset counter
            num_skipped = 0
            # stack_size = 0

            # get old stack, and update stack with current observation
            if len(frame_stack) > 0:
                old_stack = torch.cat(tuple(frame_stack), axis=0)
                stack_size, _, _ = old_stack.shape

            else:
                stack_size = 0

            update_stack(frame_stack, observation)

            if not done:
                updated_stack = torch.cat(tuple(frame_stack), axis=0)
            else:
                # when we've reached a terminal state
                updated_stack = None
            # if old stack was full, we can store transition
            if stack_size == params["stack_size"]:
                # store transition in replay buffer
                replay_buffer.push(
                    old_stack, torch.tensor([action]), updated_stack, torch.tensor([reward])
                )

            # else:
            #     num_skipped += 1

            if timestep % 100 == 0:
                optimize_dqn(replay_buffer, target_net, pred_net, optim, params)

            timestep += 1

            if len(frame_stack) == 4:
                action = select_action(
                    torch.cat(tuple(frame_stack)), timestep, pred_net, num_actions
                ).item()

            else:
                action = env.action_space.sample()
        target_net.load_state_dict(pred_net.state_dict())

    env.close()


def main():
    """
    Running the program.
    """
    # control random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    params = generate_params()

    for epoch in range(params["epochs"]):
        train(params)


if __name__ == "__main__":
    main()
