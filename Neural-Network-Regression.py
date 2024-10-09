import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Generate synthetic data
x = torch.linspace(-1, 1, 100).unsqueeze(1)  # Shape: (100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())    # Quadratic with noise, Shape: (100, 1)

# Define the neural network model
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Net, self).__init__()
        self.hidden_1 = torch.nn.Linear(input_size, hidden_sizes[0])
        self.hidden_2 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.hidden_3 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.output = torch.nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        return self.output(x)

# Model configuration
input_size = 1
hidden_sizes = [8, 16, 32]
output_size = 1
net = Net(input_size, hidden_sizes, output_size)
print(f'Model architecture:\n{net}')

# Training configuration
learning_rate = 0.03
weight_decay = 0.01
epochs = 1500
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_func = torch.nn.MSELoss()

# Plot configuration
plt.ion()
plt.figure(figsize=(12, 8))

# Training loop
for epoch in range(epochs):
    # Forward pass: compute predictions and loss
    prediction = net(x)
    loss = loss_func(prediction, y)

    # Backward pass: update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update plot every 5 epochs
    if epoch % 5 == 0:
        plt.cla()
        plt.scatter(x.numpy(), y.numpy(), label='Data')
        plt.plot(x.numpy(), prediction.detach().numpy(), 'r-', lw=2, label='Prediction')
        plt.text(0.5, 0, f'Epoch: {epoch + 1}/{epochs}\nLoss: {loss.item():.4f}', 
                 fontdict={'size': 20, 'color': 'green'})
        plt.legend()
        plt.pause(0.1)

plt.ioff()
plt.show()
