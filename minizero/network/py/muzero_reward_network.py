import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_unit import ResidualBlock, PolicyNetwork, DiscreteValueNetwork


class MuZeroRepresentationNetwork(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(MuZeroRepresentationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, num_output_channels // 2, kernel_size=3, stride=2, padding=1)
        self.residual_blocks1 = nn.ModuleList([ResidualBlock(num_output_channels // 2) for _ in range(2)])
        self.conv2 = nn.Conv2d(num_output_channels // 2, num_output_channels, kernel_size=3, stride=2, padding=1)
        self.residual_blocks2 = nn.ModuleList([ResidualBlock(num_output_channels) for _ in range(3)])
        self.avg_pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_blocks3 = nn.ModuleList([ResidualBlock(num_output_channels) for _ in range(3)])
        self.avg_pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, state):
        x = self.conv1(state)
        for residual_block in self.residual_blocks1:
            x = residual_block(x)
        x = self.conv2(x)
        for residual_block in self.residual_blocks2:
            x = residual_block(x)
        x = self.avg_pooling1(x)
        for residual_block in self.residual_blocks3:
            x = residual_block(x)
        x = self.avg_pooling2(x)
        return x


class MuZeroDynamicsNetwork(nn.Module):
    def __init__(self, num_channels, channel_height, channel_width, num_action_feature_channels, num_blocks, reward_size):
        super(MuZeroDynamicsNetwork, self).__init__()
        self.reward_size = reward_size
        self.conv = nn.Conv2d(num_channels + num_action_feature_channels, num_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.reward_network = DiscreteValueNetwork(num_channels, channel_height, channel_width, num_channels, reward_size)

    def forward(self, hidden_state, action_plane):
        x = torch.cat((hidden_state, action_plane), dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        discrete_reward = self.reward_network(x)
        return x, discrete_reward


class MuZeroPredictionNetwork(nn.Module):
    def __init__(self, num_channels, channel_height, channel_width, num_blocks, num_action_channels, action_size, num_value_hidden_channels, value_size):
        super(MuZeroPredictionNetwork, self).__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.policy = PolicyNetwork(num_channels, channel_height, channel_width, num_action_channels, action_size)
        self.value = DiscreteValueNetwork(num_channels, channel_height, channel_width, num_value_hidden_channels, value_size)

    def forward(self, hidden_state):
        x = self.conv(hidden_state)
        x = self.bn(x)
        x = F.relu(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        policy = self.policy(x)
        discrete_value = self.value(x)
        return policy, discrete_value


class MuZeroRewardNetwork(nn.Module):
    def __init__(self,
                 game_name,
                 num_input_channels,
                 input_channel_height,
                 input_channel_width,
                 num_hidden_channels,
                 hidden_channel_height,
                 hidden_channel_width,
                 num_action_feature_channels,
                 num_blocks,
                 num_action_channels,
                 action_size,
                 num_value_hidden_channels,
                 discrete_value_size):
        super(MuZeroRewardNetwork, self).__init__()
        self.game_name = game_name
        self.num_input_channels = num_input_channels
        self.input_channel_height = input_channel_height
        self.input_channel_width = input_channel_width
        self.num_hidden_channels = num_hidden_channels
        self.hidden_channel_height = hidden_channel_height
        self.hidden_channel_width = hidden_channel_width
        self.num_action_feature_channels = num_action_feature_channels
        self.num_blocks = num_blocks
        self.num_action_channels = num_action_channels
        self.action_size = action_size
        self.num_value_hidden_channels = num_value_hidden_channels
        self.discrete_value_size = discrete_value_size

        self.representation_network = MuZeroRepresentationNetwork(num_input_channels, num_hidden_channels)
        self.dynamics_network = MuZeroDynamicsNetwork(num_hidden_channels, hidden_channel_height, hidden_channel_height, num_action_feature_channels, num_blocks, discrete_value_size)
        self.prediction_network = MuZeroPredictionNetwork(num_hidden_channels, hidden_channel_height, hidden_channel_width, num_blocks,
                                                          num_action_channels, action_size, num_value_hidden_channels, discrete_value_size)

    @torch.jit.export
    def get_type_name(self):
        return "muzero_reward"

    @torch.jit.export
    def get_game_name(self):
        return self.game_name

    @torch.jit.export
    def get_num_input_channels(self):
        return self.num_input_channels

    @torch.jit.export
    def get_input_channel_height(self):
        return self.input_channel_height

    @torch.jit.export
    def get_input_channel_width(self):
        return self.input_channel_width

    @torch.jit.export
    def get_num_hidden_channels(self):
        return self.num_hidden_channels

    @torch.jit.export
    def get_hidden_channel_height(self):
        return self.hidden_channel_height

    @torch.jit.export
    def get_hidden_channel_width(self):
        return self.hidden_channel_width

    @torch.jit.export
    def get_num_action_feature_channels(self):
        return self.num_action_feature_channels

    @torch.jit.export
    def get_num_blocks(self):
        return self.num_blocks

    @torch.jit.export
    def get_num_action_channels(self):
        return self.num_action_channels

    @torch.jit.export
    def get_action_size(self):
        return self.action_size

    @torch.jit.export
    def get_num_value_hidden_channels(self):
        return self.num_value_hidden_channels

    @torch.jit.export
    def get_discrete_value_size(self):
        return self.discrete_value_size

    @torch.jit.export
    def initial_inference(self, state):
        # representation + prediction
        hidden_state = self.representation_network(state)
        hidden_state = self.scale_hidden_state(hidden_state)
        policy, discrete_value = self.prediction_network(hidden_state)
        value = self.discrete_value_to_value_scalar(discrete_value)
        return {"policy": policy, "value": value, "discrete_value": discrete_value, "hidden_state": hidden_state}

    @torch.jit.export
    def recurrent_inference(self, hidden_state, action_plane):
        # dynamics + prediction
        next_hidden_state, discrete_reward = self.dynamics_network(hidden_state, action_plane)
        next_hidden_state = self.scale_hidden_state(next_hidden_state)
        policy, discrete_value = self.prediction_network(next_hidden_state)
        reward = self.discrete_value_to_value_scalar(discrete_reward)
        value = self.discrete_value_to_value_scalar(discrete_value)
        return {"policy": policy, "value": value, "discrete_value": discrete_value, "reward": reward, "discrete_reward": discrete_reward, "hidden_state": next_hidden_state}

    def scale_hidden_state(self, hidden_state):
        # scale hidden state to range [0, 1] for each feature plane
        batch_size, channel, _, _ = hidden_state.shape
        min_val = hidden_state.min(-1).values.min(-1).values.view(batch_size, channel, 1, 1)
        max_val = hidden_state.max(-1).values.max(-1).values.view(batch_size, channel, 1, 1)
        scale = (max_val - min_val)
        scale[scale < 1e-5] += 1e-5
        hidden_state = (hidden_state - min_val) / scale
        return hidden_state

    def discrete_value_to_value_scalar(self, discrete_value):
        discrete_value_prob = torch.softmax(discrete_value, dim=1)
        value_range = torch.tensor([v for v in range(self.discrete_value_size // 2, self.discrete_value_size // 2 + 1)]).expand(discrete_value.size(0), -1).to(device=discrete_value_prob.device)
        value = (discrete_value_prob * value_range).sum(1, keepdim=True)
        # invert
        epsilon = 0.001
        value = torch.sign(value) * (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon)) ** 2 - 1)
        return value

    def forward(self, state, action_plane=torch.empty(0)):
        if action_plane.numel() == 0:
            return self.initial_inference(state)
        else:
            return self.recurrent_inference(state, action_plane)
