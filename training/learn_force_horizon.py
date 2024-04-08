import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os
import glob
import rosbag
from geometry_msgs.msg import WrenchStamped, PoseStamped
from sensor_msgs.msg import JointState
from network_structures.RNN_horizon import DynamicsModelRNN

def read_bagfile(bag_file_path):
    """
    Reads specified ROS topics from a .bag file and converts them to numpy arrays.
    Note: Adapt this function to match the actual structure of your ROS messages.
    """
    f_ee = []  # Force exerted by the end effector
    states = []  # Robot's states (e.g., joint angles)
    actions = []  # Actions taken (could be velocities or other command signals)

    bag = rosbag.Bag(bag_file_path)
    for topic, msg, t in bag.read_messages(topics=['/contact_wrench', '/franka_joint_angle', '/obj_pose']):
        if topic == '/contact_wrench':
            f_ee.append([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
            # f_ee.append([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                        #  msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        elif topic == '/franka_joint_angle':
            actions.append(list(msg.joint_angle))  # Modify according to your definition of action
        elif topic == '/obj_pose':
            states.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                                msg.pose.orientation.x, msg.pose.orientation.y,
                                msg.pose.orientation.z, msg.pose.orientation.w])
    bag.close()

    # Convert to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    f_ee = np.array(f_ee)
    # next_states = np.roll(states, -1, axis=0)  # Assuming next state is the subsequent state; adjust as needed

    return states, actions, f_ee#, next_states[:-1]  # Exclude the last element as it has no next state

def prepare_sequences(states, actions, f_ee, sequence_length=50):
    """
    Prepare data sequences for training the model to predict a sequence of future states.
    This function assumes that `states`, `actions`, and `next_states` are numpy arrays
    containing sequences of data points.
    """
    X, y = [], []
    for i in range(len(states) - sequence_length):
        state_sequence = states[i]  # Initial state
        # bolt_state_sequence = bolt_states[i] # Initial bolt state
        f_ee_sequence = f_ee[i] # Initial f_ee state
        action_sequence = actions[i:i+sequence_length].flatten()  # Sequence of actions
        next_f_ee_sequence = f_ee[i+1:i+1+sequence_length].flatten()
        X.append(np.concatenate([state_sequence, f_ee_sequence, action_sequence]))
        y.append(np.concatenate([next_f_ee_sequence]))
    return np.array(X), np.array(y)

def main():
    # Load data from .bag files instead of .npz
    data_dir = "data"  # Update this path
    bag_files = glob.glob(os.path.join(data_dir, "*.bag"))
    all_states, all_actions, all_f_ee = [], [], []
    for bag_file_path in bag_files:
        states, actions, f_ee = read_bagfile(bag_file_path)
        all_states.append(states)
        all_actions.append(actions)
        all_f_ee.append(f_ee)
        # all_bolt_states.append(bolt_states)
        # all_next_states.append(next_states)

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    f_ee = np.concatenate(all_f_ee, axis=0)
    # bolt_states = np.concatenate(all_bolt_states, axis=0)
    # next_states = np.concatenate(all_next_states, axis=0)

    # Prepare sequences
    sequence_length = 50
    X, y = prepare_sequences(states, actions, f_ee, sequence_length)
    
    # Model dimensions
    input_dim = X.shape[1]  # Adjusted automatically based on the data
    output_dim = y.shape[1]  # Automatically determined
    hidden_dim = 128

    # Model initialization
    model = DynamicsModelRNN(input_dim, hidden_dim, output_dim)
    
    # Split and scale data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to PyTorch datasets
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    # DataLoader setup
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    epochs = 200
    
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                predictions = model(X_val)
                val_loss += criterion(predictions, y_val).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}')
    
    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Save the model and scaler
    model_save_path = "contact_force_model_horizon_full_state.pth"  # Update path as needed
    scaler_save_path = "scaler_contact_force_horizon_full_state.save"  # Update path as needed
    torch.save(model.state_dict(), model_save_path)
    joblib.dump(scaler, scaler_save_path)

if __name__ == "__main__":
    main()
