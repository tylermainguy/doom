import math
import random
from collections import deque, namedtuple

import gym
import torch
import torch.nn.functional as F
import vizdoomgym
from torch.utils.tensorboard import SummaryWriter
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


class Trainer:
    """Train DQN model."""

    def __init__(self, params):

        self.params = params

        # Initialize the environment
        self.env = gym.make("VizdoomBasic-v0")
        self.num_actions = self.env.action_space.n

        # Intitialize both deep Q networks
        self.target_net = DQN(60, 80, num_actions=self.num_actions)
        self.pred_net = DQN(60, 80, num_actions=self.num_actions)

        # Create env, initialize the starting state
        self.target_net.load_state_dict(self.pred_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), 3e-4)

        if self.params["device"] == "cuda":

            self.target_net = self.target_net.to(self.params["device"])
            self.pred_net = self.pred_net.to(self.params["device"])
        # Initialize replay memory
        self.replay_buffer = ReplayBuffer(50000)

        # Initialize frame stack
        self.stack_size = params["stack_size"]
        self.frame_stack = deque(maxlen=self.stack_size)
        self.stack_rewards = deque(maxlen=self.stack_size)

        self.losses = AverageMeter()

        self.writer = SummaryWriter()

    @torch.enable_grad()
    def train(self, epoch):
        """Run a single epoch of training."""
        self.target_net.eval()

        frames = 0
        steps = 0

        if self.params["load_model"]:
            self.pred_net.load_state_dict(
                torch.load("model.pk", map_location=torch.device(self.params["device"]))
            )
            self.target_net.load_state_dict(
                torch.load("model.pk", map_location=torch.device(self.params["device"]))
            )

        for episode in tqdm(range(self.params["episodes"]), desc="episodes", unit="episodes"):

            done = False
            self.env.reset()

            # For frame skipping
            num_skipped = 0
            skipped_rewards = 0

            action = self.env.action_space.sample()

            self.reset_stack()

            while not done:
                frames += 1
                self.env.render()
                observation, reward, done, _ = self.env.step(action)
                skipped_rewards += reward

                # only want to stack every four frames
                if num_skipped == self.params["skip_frames"] - 1:

                    # reset counter
                    num_skipped = 0

                    # want average reward over skipped frames
                    skipped_rewards = float(skipped_rewards) / self.params["skip_frames"]

                    # get old stack, and update stack with current observation
                    if len(self.frame_stack) > 0:
                        old_stack = torch.cat(tuple(self.frame_stack), axis=0)
                        curr_size, _, _ = old_stack.shape

                    else:
                        curr_size = 0

                    self.update_stack(observation, reward)
                    skipped_rewards = 0

                    if not done:
                        updated_stack = torch.cat(tuple(self.frame_stack), axis=0)
                    else:
                        # when we've reached a terminal state
                        updated_stack = None

                    # if old stack was full, we can store transition
                    if curr_size == self.params["stack_size"]:
                        total_reward = self.get_stack_sum()
                        # store transition in replay buffer
                        self.replay_buffer.push(
                            old_stack,
                            torch.tensor([action]),
                            updated_stack,
                            torch.tensor([total_reward]),
                        )

                    # if we can select action using frame stack
                    if len(self.frame_stack) == 4:
                        action = self.select_action(
                            torch.cat(tuple(self.frame_stack)), steps, self.num_actions
                        ).item()
                        steps += 1

                    # optimize network every 100 timesteps
                    if steps % 20 == 0:
                        self.optimize_dqn()
                        self.writer.add_scalar(
                            "loss", self.losses.avg, epoch * self.params["episodes"] + episode
                        )

                    if steps % 10000 == 0:
                        self.target_net.load_state_dict(self.pred_net.state_dict())
                        torch.save(self.pred_net.state_dict(), "model.pk")

                else:
                    num_skipped += 1

        self.env.close()

    @torch.no_grad()
    def evaluate(self):
        """Visually evaluate models performance."""

        self.pred_net.load_state_dict(
            torch.load("model.pk", map_location=torch.device(self.params["device"]))
        )
        self.target_net.load_state_dict(
            torch.load("model.pk", map_location=torch.device(self.params["device"]))
        )

        steps = 0
        for episode in tqdm(range(self.params["episodes"]), desc="episodes", unit="episodes"):

            done = False
            self.env.reset()

            # For frame skipping
            num_skipped = 0
            skipped_reward = 0

            action = self.env.action_space.sample()

            self.reset_stack()

            while not done:
                self.env.render()
                observation, reward, done, _ = self.env.step(action)

                skipped_reward += reward

                # only want to stack every four frames
                if num_skipped == self.params["skip_frames"] - 1:

                    # reset counter
                    num_skipped = 0

                    # get old stack, and update stack with current observation
                    if len(self.frame_stack) > 0:
                        old_stack = torch.cat(tuple(self.frame_stack), axis=0)
                        curr_size, _, _ = old_stack.shape

                    else:
                        curr_size = 0

                    skipped_reward = float(skipped_reward) / self.params["skip_frames"]
                    self.update_stack(observation, skipped_reward)
                    skipped_reward = 0

                    if not done:
                        updated_stack = torch.cat(tuple(self.frame_stack), axis=0)
                    else:
                        # when we've reached a terminal state
                        updated_stack = None

                    # if we can select action using frame stack
                    if len(self.frame_stack) == 4:
                        action = self.select_action(
                            torch.cat(tuple(self.frame_stack)), steps, self.num_actions
                        ).item()
                        steps += 1

                else:
                    num_skipped += 1

    def reset_stack(self):
        """reset frame stack."""
        self.frame_stack = deque(maxlen=self.stack_size)

    def update_stack(self, observation, reward):
        """
        Update the frame stack with the an observation
        This function will create stacks of four consequtive
        images for the model to learn from.
        """
        image = self.preprocess(observation)

        self.frame_stack.append(image)
        self.stack_rewards.append(reward)

    def get_stack_sum(self):
        total = 0

        for reward in self.stack_rewards:
            total += reward

        return total

    def preprocess(self, observation):
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

    def select_action(self, state, timesteps, num_actions):
        """
        Select_action uses a epsilon greedy method to select and action.
        This involves both greedy action selection and random action
        selection for exploration of our agent.
        """
        # threshold for exploration
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * timesteps / EPS_DECAY)

        # exploit
        if sample > eps_threshold:
            with torch.no_grad():
                # choose action with highest q-value
                state = state.unsqueeze(0).to(self.params["device"])
                return self.pred_net(state).max(1)[1].view(1, 1)
        # explore
        else:
            return torch.tensor([[random.randrange(num_actions)]], dtype=torch.long)

    def optimize_dqn(self):
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
        if len(self.replay_buffer) < self.params["batch_size"] * 2:
            return

        # sample batch, and transpose batch
        sample = self.replay_buffer.sample(self.params["batch_size"])
        batch = Transition(*zip(*sample))

        # mask any transitions that have None (indicate transition to terminal state)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
        ).to(self.params["device"])
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(
            self.params["device"]
        )

        # get batches of states, actions, and rewards for DQN
        state_batch = torch.stack(batch.state).to(self.params["device"])
        action_batch = torch.stack(batch.action).to(self.params["device"])
        reward_batch = torch.stack(batch.reward).to(self.params["device"])

        # Calculate the Q-value of the current state-action pair using the model.
        state_action_values = self.pred_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.params["batch_size"]).to(self.params["device"])

        # find values associated with non-final transitions
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )

        # calculate expected value using non-terminal transition values
        expected_state_action_values = (next_state_values * GAMMA).unsqueeze(1) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.losses.update(loss.item(), self.params["batch_size"])

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.pred_net.parameters(), 1)

        self.optimizer.step()
