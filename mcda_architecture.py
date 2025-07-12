import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import random

# --- Layer 3 (Base): Complete CDA Components ---

class I_Neuron:
    """The basic Leaky Integrate-and-Fire processing neuron."""
    def __init__(self, tau=20.0, v_threshold=1.0):
        self.tau = tau
        self.v_threshold = v_threshold
        self.potential = torch.tensor(0.0)

    def update(self, input_current, dt=1.0):
        leak = -self.potential / self.tau
        self.potential += (leak + input_current) * dt
        if self.potential >= self.v_threshold:
            self.potential = torch.tensor(0.0)
            return torch.tensor(1.0)
        return torch.tensor(0.0)

class M_Neuron(nn.Module):
    """The Modulator Neuron that learns to control structural plasticity."""
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
        self.policy_network = nn.Sequential(
            nn.Linear(num_neurons * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_neurons * num_neurons * 2)
        )

    def forward(self, presynaptic_spikes, postsynaptic_spikes):
        combined_activity = torch.cat((presynaptic_spikes, postsynaptic_spikes))
        policy_output = self.policy_network(combined_activity)
        pruning_signal, genesis_signal = torch.chunk(policy_output, 2)
        pruning_matrix = torch.sigmoid(pruning_signal.view(self.num_neurons, self.num_neurons))
        genesis_matrix = torch.sigmoid(genesis_signal.view(self.num_neurons, self.num_neurons))
        return pruning_matrix, genesis_matrix

class CDA_Subnet(nn.Module):
    """The advanced CDA Subnet, controlled by an M_Neuron."""
    def __init__(self, num_inputs, num_neurons, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.i_neurons = [I_Neuron() for _ in range(num_neurons)]
        self.m_neuron = M_Neuron(num_neurons)
        self.input_weights = nn.Parameter(torch.randn(num_inputs, num_neurons) * 0.1)
        self.recurrent_weights = nn.Parameter(torch.rand(num_neurons, num_neurons) * 0.05)
        self.recurrent_weights.data.fill_diagonal_(0)

    def forward(self, input_spikes, prev_spikes):
        total_current = torch.matmul(input_spikes, self.input_weights) + torch.matmul(prev_spikes, self.recurrent_weights)
        current_spikes = torch.tensor([n.update(total_current[i]) for i, n in enumerate(self.i_neurons)])
        pruning_matrix, genesis_matrix = self.m_neuron(prev_spikes, current_spikes)
        self._apply_structural_plasticity(pruning_matrix, genesis_matrix)
        return current_spikes

    def _apply_structural_plasticity(self, pruning_matrix, genesis_matrix):
        self.recurrent_weights.data -= self.learning_rate * pruning_matrix
        self.recurrent_weights.data += self.learning_rate * genesis_matrix
        self.recurrent_weights.data = torch.clamp(self.recurrent_weights.data, min=0)
        self.recurrent_weights.data.fill_diagonal_(0)

class CDA_Layer(nn.Module):
    """The full CDA Layer, composed of multiple advanced subnets."""
    def __init__(self, num_subnets, num_inputs, num_neurons_per_subnet):
        super().__init__()
        self.subnets = nn.ModuleList([
            CDA_Subnet(num_inputs, num_neurons_per_subnet) for _ in range(num_subnets)
        ])
    def forward(self, focused_input, prev_subnet_outputs):
        all_outputs = [subnet(focused_input, prev_subnet_outputs[i]) for i, subnet in enumerate(self.subnets)]
        return all_outputs

# --- Layer 2: Global Workspace Module (GWM) ---
class GlobalWorkspaceModule:
    def attend(self, list_of_inputs, action_index):
        return list_of_inputs[action_index]

# --- Layer 1 (Core): Advanced IQ Components ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(nn.Linear(state_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_size))
    def forward(self, state): return self.network(state)

class ReplayBuffer:
    def __init__(self, capacity): self.buffer = collections.deque(maxlen=capacity)
    def add(self, s, a, r, s_next, d): self.buffer.append((s, a, r, s_next, d))
    def sample(self, batch_size): return random.sample(self.buffer, batch_size)
    def __len__(self): return len(self.buffer)

class IQ_Layer:
    def __init__(self, state_size, action_size, learning_rate=1e-4, gamma=0.99):
        self.state_size=state_size; self.action_size=action_size; self.gamma=gamma
        self.policy_net=DQN(state_size, action_size); self.target_net=DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer=optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn=nn.MSELoss(); self.memory=ReplayBuffer(10000)
    def _state_to_tensor(self, state):
        s_vec=np.zeros(self.state_size); s_vec[state]=1.0
        return torch.FloatTensor(s_vec).unsqueeze(0)
    def choose_action(self, state, epsilon=0.1):
        if random.random()<epsilon: return random.randrange(self.action_size)
        with torch.no_grad():
            q_values=self.policy_net(self._state_to_tensor(state))
            return torch.argmax(q_values).item()
    def train_from_memory(self, batch_size=64):
        if len(self.memory)<batch_size: return None
        states,acts,rews,next_sts,dones=zip(*self.memory.sample(batch_size))
        s_batch=torch.zeros(batch_size,self.state_size); next_s_batch=torch.zeros(batch_size, self.state_size)
        for i in range(batch_size): s_batch[i][states[i]]=1.0; next_s_batch[i][next_sts[i]]=1.0
        a_batch=torch.LongTensor(acts).unsqueeze(1); r_batch=torch.FloatTensor(rews).unsqueeze(1); d_batch=torch.BoolTensor(dones).unsqueeze(1)
        q_vals=self.policy_net(s_batch).gather(1, a_batch)
        with torch.no_grad():
            next_q=self.target_net(next_s_batch).max(1)[0].unsqueeze(1); next_q[d_batch]=0.0
        target_q=r_batch+(self.gamma*next_q); loss=self.loss_fn(q_vals,target_q)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return loss.item()

# --- The Final Orchestrator ---
class MCDA:
    """The complete and final Meta-Cognitive Dynamic Architecture."""
    def __init__(self, state_size, num_actions, num_inputs, num_subnets, neurons_per_subnet):
        self.iq_layer = IQ_Layer(state_size, num_actions)
        self.gwm = GlobalWorkspaceModule()
        self.cda_layer = CDA_Layer(num_subnets, num_inputs, neurons_per_subnet)
        self.prev_cda_outputs = [torch.zeros(neurons_per_subnet) for _ in range(num_subnets)]

    def step(self, list_of_inputs, state, epsilon):
        # 1. IQ Layer chooses an action
        action = self.iq_layer.choose_action(state, epsilon)
        
        # 2. GWM attends to the chosen input
        focused_input = self.gwm.attend(list_of_inputs, action)
        
        # 3. CDA Layer processes the focused input
        new_cda_outputs = self.cda_layer(focused_input, self.prev_cda_outputs)
        self.prev_cda_outputs = new_cda_outputs # Update recurrent state
        
        final_output_vector = torch.cat(new_cda_outputs)
        
        return action, final_output_vector
        
    def get_current_state(self, cda_output):
        return 1 if torch.sum(cda_output).item() > 5 else 0
