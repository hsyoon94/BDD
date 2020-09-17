import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

expert_demo_file = torch.load('./expert_demo/data-20200721-200.pt')
expert_demo_states = expert_demo_file['states'].numpy()
# shape 60000 * 8
expert_demo_states = np.reshape(expert_demo_states, [expert_demo_states.shape[0] * expert_demo_states.shape[1], expert_demo_states.shape[2]])

states = np.array([])
actions = np.array([])
state_length = 6
action_length = 3

for i in range(expert_demo_states.shape[0]):
    if expert_demo_states.shape[0] - i > action_length:
        state = expert_demo_states[i][2:]
        action = np.array([])
        for j in range(action_length):
            action = np.append(action, expert_demo_states[i + 1 + j][0:2])

        if i % 10000 == 0:
            print("step", i)

        states = np.append(states, state)
        actions = np.append(actions, action)

states = np.reshape(states, [int(states.shape[0]/state_length), state_length])
actions = np.reshape(actions, [int(actions.shape[0]/(action_length*2)), (action_length*2)])

print(states.shape)
print(actions.shape)

data = {
    'states': states,
    'actions': actions
}

save_dir = './expert_demo/expert_demo_test_' + str(action_length) + '.pt'
torch.save(data, save_dir)