# ---------------------------------------------------------
# Author: Dr Pantelis Georgiades
#         Computation-based Science and Technology Resarch
#         Centre (CaSToRC) - The Cyprus Institute
# License: MIT
# ---------------------------------------------------------

import torch
import numpy as np

import torch.optim as optim

# ---------------------------------------------------------

from .utils import load_configs

# ---------------------------------------------------------
def train(model, dataset, config=None):
    # If the user enters the path to the config load it else raise error
    if isinstance(config, str):
        config = load_configs(config)
    elif config is None:
        raise ValueError("ValueError: No config file found.")
    # If there is an integer seed defined in the config set it for torch
    if isinstance(config['torch']['seed'], int):
        print(f"Setting torch seed to {config['torch']['seed']}")
        torch.manual_seed(config['torch']['seed'])
    # Get the number of epochs and learning rate from the config
    epochs = config['training']['epochs']
    lr = config['training']['lr']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.cuda()

    # Optimizer and Criterion
    optimizer = optim.Adam(params=model.parameters(), lr = lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    for epoch in np.arange(1, epochs, 1):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in dataset:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            acc = ((output.argmax(dim=1) == label).float().mean())
            epoch_accuracy += acc/len(dataset)
            epoch_loss += loss/len(dataset)
        
        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch+1, epoch_accuracy, epoch_loss))