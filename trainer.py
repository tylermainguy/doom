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
from replay_memory import ReplayMemory

# Initialize tranisiton memory  tuple
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


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

        self.pred_net.apply(init_weights)

        # Create env, initialize the starting state
        self.target_net.load_state_dict(self.pred_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), 3e-4)

        if self.params["device"] == "cuda":

            self.target_net = self.target_net.to(self.params["device"])
            self.pred_net = self.pred_net.to(self.params["device"])
        # Initialize replay memory
        self.replay_memory = ReplayMemory(50000)

        # Initialize frame stack
        self.stack_size = params["stack_size"]
        self.frame_stack = deque(maxlen=self.stack_size)

        self.losses = AverageMeter()

        self.writer = SummaryWriter()

        self.steps = 0
        self.learning_steps = 0
        self.epsilon = self.params["eps_start"]
        self.epsilon_start = self.params["eps_start"]

    @torch.enable_grad()
    def train(self):
        """Run a single epoch of training."""
        self.target_net.eval()

        frames = 0
        self.steps = 0

        if self.params["load_model"]:
            self.pred_net.load_state_dict(
                torch.load("model.pk", map_location=torch.device(self.params["device"]))
            )
            self.target_net.load_state_dict(
                torch.load("model.pk", map_location=torch.device(self.params["device"]))
            )

        pbar = tqdm(range(self.params["episodes"]), unit="episodes")
        for episode in pbar:
            pbar.set_description("episode: {} ls: {}".format(episode, self.learning_steps))

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

                if action == 2 and reward < 0:
                    reward = -10.0
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

                    self.update_stack(observation)

                    if not done:
                        updated_stack = torch.cat(tuple(self.frame_stack), axis=0)
                    else:
                        # when we've reached a terminal state
                        updated_stack = None

                    # if old stack was full, we can store transition
                    if curr_size == self.params["stack_size"]:
                        # store transition in replay buffer

                        self.replay_memory.add_memory(
                            old_stack,
                            torch.tensor([action]),
                            updated_stack,
                            torch.tensor([skipped_rewards]),
                        )

                    skipped_rewards = 0
                    # if we can select action using frame stack
                    if len(self.frame_stack) == 4:
                        action = self.select_action(
                            torch.cat(tuple(self.frame_stack)), self.num_actions
                        ).item()
                        self.steps += 1

                    self.train_dqn()

                    if self.steps % 200 == 0:
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

            action = self.env.action_space.sample()

            self.reset_stack()

            while not done:
                self.env.render()
                observation, reward, done, _ = self.env.step(action)

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

                    self.update_stack(observation)

                    if not done:
                        updated_stack = torch.cat(tuple(self.frame_stack), axis=0)
                    else:
                        # when we've reached a terminal state
                        updated_stack = None

                    # if we can select action using frame stack
                    if len(self.frame_stack) == 4:
                        action = self.select_action(
                            torch.cat(tuple(self.frame_stack)), self.num_actions
                        ).item()
                        steps += 1

                else:
                    num_skipped += 1

    def reset_stack(self):
        """reset frame stack."""
        self.frame_stack = deque(maxlen=self.stack_size)

    def update_stack(self, observation):
        """
        Update the frame stack with the an observation
        This function will create stacks of four consequtive
        images for the model to learn from.
        """
        image = self.preprocess(observation)

        self.frame_stack.append(image)

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

    def select_action(self, state, num_actions):
        """
        Select_action uses a epsilon greedy method to select and action.
        This involves both greedy action selection and random action
        selection for exploration of our agent.
        """
        # threshold for exploration
        sample = random.random()

        self.epsilon_decay()
        # exploit
        if sample > self.epsilon:
            with torch.no_grad():
                # choose action with highest q-value
                state = state.unsqueeze(0).to(self.params["device"])
                return self.pred_net(state).max(1)[1].view(1, 1)
        # explore
        else:
            return torch.tensor([[random.randrange(num_actions)]], dtype=torch.long)

    def epsilon_decay(self):
        if self.learning_steps < 100000 or self.learning_steps > 200000:
            return

        decay_rate = (self.params["eps_start"] - self.params["eps_end"]) / 100000.0
        self.epsilon = self.params["eps_start"] - decay_rate * self.learning_steps

    def train_dqn(self):
        """
        Perform optimization on the DQN given a batch of randomly
        sampled transitions from the replay buffer.
        """

        # need to be able to sample
        if len(self.replay_memory) < self.replay_memory.max_size:
            return

        self.learning_steps += 1

        batch = self.replay_memory.sample(self.params["batch_size"])

        new_states = [x[2] for x in batch]

        # can't transition to None (termination)
        non_terminating_filter = torch.tensor(
            tuple(map(lambda s: s is not None, new_states)), device=self.params["device"],
        )
        non_terminating = torch.stack([s for s in new_states if s is not None]).to(
            self.params["device"]
        )

        states = torch.stack([x[0] for x in batch]).to(self.params["device"])
        actions = torch.stack([x[1] for x in batch]).to(self.params["device"])
        rewards = torch.stack([x[3] for x in batch]).to(self.params["device"])

        with torch.no_grad():
            predicted = self.pred_net(states)
            action_val = predicted.gather(1, actions)

        target_vals = torch.zeros(self.params["batch_size"], device=self.params["device"])

        # # calculate maximum action for new states
        target_vals[non_terminating_filter] = self.target_net(non_terminating).max(dim=1)[0]
        target_vals = target_vals.unsqueeze(1)

        target_update = (self.params["gamma"] * target_vals) + rewards

        huber_loss = F.smooth_l1_loss(action_val, target_update)

        self.optimizer.zero_grad()

        huber_loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.pred_net.parameters(), 1)

        self.optimizer.step()
