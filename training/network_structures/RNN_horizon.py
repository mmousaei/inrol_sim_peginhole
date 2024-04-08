from torch import nn
import torch
# from torch.utils.tensorboard import SummaryWriter
# from torchviz import make_dot

class DynamicsModelRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(DynamicsModelRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Assuming input_dim considers the concatenated state and action sequences as input
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Output layer that maps from hidden state space to output state space
        # Adjusted to output a sequence of the desired length
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x expected to be of shape (batch, sequence_length, features)
        # No need to check for a time dimension here, as we expect a sequence
        lstm_out, _ = self.lstm(x)
        # Apply linear layer to each time step
        output = self.linear(lstm_out)
        return output

if __name__ == "__main__":
    # Instantiate the model
    input_dim = 12+6+50*6  # Example input dimension
    hidden_dim = 128  # Example hidden dimension
    output_dim = 50*3  # Example output dimension
    model = DynamicsModelRNN(input_dim, hidden_dim, output_dim)

    model.eval()

    seq_length = 3
    feature_size = 12 + 6 + 50*6  # Calculate based on your model's expected input dimensions
    batch_size = 1  # Keeping it as 1 for the dummy input

    # # Create a new dummy input with the correct dimensions
    # dummy_input = torch.randn(batch_size, seq_length, feature_size)

    # # Now, try exporting the model again with the adjusted dummy input
    # torch.onnx.export(model,               # model being run
    #                 dummy_input,         # model input (or a tuple for multiple inputs)
    #                 "model.onnx",        # where to save the model (can be a file or file-like object)
    #                 export_params=True,  # store the trained parameter weights inside the model file
    #                 opset_version=10,    # the ONNX version to export the model to
    #                 do_constant_folding=True,  # whether to execute constant folding for optimization
    #                 input_names=['input'],   # the model's input names
    #                 output_names=['output'], # the model's output names
    #                 dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
    #                                 'output': {0: 'batch_size'}})
    x = torch.randn(batch_size, seq_length, feature_size)  # Example input tensor

    # Visualizing the model
    y = model(x)
    vis_graph = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    vis_graph.render('DynamicsModelRNN', format='png')  # Saves the visualization as a PNG image