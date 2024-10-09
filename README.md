# PyTorch Regression Example

This project demonstrates a simple regression task using PyTorch. The model is trained to approximate a quadratic function with added noise, using a neural network with three hidden layers. The code visualizes the training process by plotting the data points and the regression line at each epoch.

## Project Structure

- `main.py`: The main script that contains the code for the regression model, training loop, and visualization.

## Requirements

- Python 3.x
- PyTorch
- Matplotlib

You can install the required packages using:

```bash
pip install torch matplotlib
```

## Model Architecture
The neural network model used in this example has the following structure:

- Input Layer: 1 feature
- Hidden Layer 1: 8 neurons, ReLU activation
- Hidden Layer 2: 16 neurons, ReLU activation
- Hidden Layer 3: 32 neurons, ReLU activation
- Output Layer: 1 output

# Training Details
- Optimizer: Stochastic Gradient Descent (SGD)
- Learning Rate: 0.03
- Weight Decay: 0.01 (L2 regularization)
- Loss Function: Mean Squared Error (MSE)
- Epochs: 1500
The training process is visualized, with the regression line updated every 5 epochs.

How to Run the Code
1- Clone this repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
```
2- Navigate to the project directory:
```bash
cd your-repo-name
```
3- Run the script:
```bash
python main.py
```
This will train the model and display a plot showing the regression line fitting the noisy data over time.

# Visualization
The script uses Matplotlib to visualize the learning process. It plots the original data points and the model's predictions, updating the plot every 5 epochs. The plot includes the current epoch and the loss value displayed as text.

# Example Output
https://github.com/user-attachments/assets/d8c05683-5709-4acc-a544-a6b8307f6cc1
