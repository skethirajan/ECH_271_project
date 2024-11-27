"""This module contains the main functions for needed to train a model on the MOF-Galaxy dataset.
The main functions are:
- Create a dataset from the MOF-Galaxy dataset based on the values of
    alpha (similarity weighting parameter) and omega parameters (edge weight threshold).
- Computes the mean aggregates along all the features of the neighbors.
    This is done for all MOFs. Note that the MOF itself is also included in the mean aggregate.
- Converts the input numpy arrays to torch tensors (needed for PyTorch).
- Splits the dataset into train, validation and test sets.
- Tains the model and returns the trained model and the metrics (loss & accuracy).
- Plots the results of the trained model
"""

import pathlib
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from torch.optim import Adam, lr_scheduler
import pandas as pd
import numpy as np
import networkx as nx


torch.manual_seed(271)

def load_dataframes(alpha, omega):
    """Loads the dataframes from the csv files.
    If the edge list does not exist, it is created and saved.
    """

    if alpha > 1 or omega > 1 or alpha < 0 or omega < 0:
        raise ValueError("alpha and omega must be between 0 and 1")

    parent_data_path = pathlib.Path(__file__).parent.resolve().joinpath("data")
    data = pd.read_csv(parent_data_path.joinpath("SMILES_METAL_1988_NoPLD.csv"))
    edge_path = parent_data_path.joinpath(f"EdgesList_1988_{alpha}_alpha_{omega}_omega.csv")

    if not edge_path.exists():
        linker_sim_path = parent_data_path.joinpath("similarity/linkers_similarity.npy")
        mof_sim_path = parent_data_path.joinpath("similarity/mof_features_similarity.npy")
        linker_sim_arr = np.load(linker_sim_path)
        mof_sim_arr = np.load(mof_sim_path)
        adj_arr = alpha*linker_sim_arr + (1-alpha)*mof_sim_arr
        indices = np.argwhere(adj_arr < omega)
        for index in indices:
            adj_arr[*index] = 0.0
        np.fill_diagonal(adj_arr, 0.0)
        G = nx.from_numpy_array(adj_arr, create_using=nx.DiGraph)
        edge_pd = nx.to_pandas_edgelist(G)
        edge_pd.to_csv(edge_path, index=False)

    edge_pd = pd.read_csv(edge_path)
    return data, edge_pd

def mean_aggregate(mofA_id, edge_pd, data):
    """Calculates the mean aggregate of the neighbors of a given MOF."""

    neighbors_id = edge_pd.loc[edge_pd["source"] == mofA_id]["target"].to_numpy()
    neighbors_features = data.iloc[neighbors_id, 0:6].to_numpy()
    mean_aggregate = (np.sum(neighbors_features, axis=0) + data.iloc[mofA_id, 0:6].to_numpy()) / (len(neighbors_id) + 1)
    return mean_aggregate

def mean_aggregate_all(x_arr, edge_pd, data):
    """Calculates the mean aggregate of all MOFs."""

    x_agg_arr = np.zeros(x_arr.shape)
    for i, mofA_id in enumerate(range(x_agg_arr.shape[0])):
        x_agg_arr[i] = mean_aggregate(mofA_id, edge_pd, data)
    return x_agg_arr

def convert_to_tensors(x_arr, x_agg_arr, y_arr, edge_arr, edge_weights_arr):
    """Converts the numpy arrays to torch tensors."""

    x = torch.tensor(x_arr).float()
    x_agg = torch.tensor(x_agg_arr).float()
    y = torch.tensor(y_arr)
    edge_index = torch.tensor(edge_arr).long()
    edge_attr = torch.tensor(edge_weights_arr)
    return x, x_agg, y, edge_index, edge_attr

def extract_mof_data(data, edge_pd):
    """Extracts the MOF data from the dataframes."""

    nodes_arr = data.iloc[:, 6].to_numpy()
    x_arr = data.iloc[:,0:6].to_numpy()
    y_arr = data.iloc[:,-1].to_numpy()
    edge_arr = edge_pd.iloc[:,0:-1].to_numpy().T
    edge_weights_arr = edge_pd.iloc[:,-1].to_numpy()
    x_agg_arr = mean_aggregate_all(x_arr, edge_pd, data)
    return nodes_arr, x_arr, x_agg_arr, y_arr, edge_arr, edge_weights_arr

def get_data(device, num_val=0.2, num_test=0.0, alpha=0.9, omega=0.9):
    """Loads the data and splits it into train, validation and test sets."""

    data, edge_pd = load_dataframes(alpha, omega)
    nodes_arr, x_arr, x_agg_arr, y_arr, edge_arr, edge_weights_arr = extract_mof_data(data, edge_pd)
    x, x_agg, y, edge_index, edge_attr = convert_to_tensors(x_arr, x_agg_arr, y_arr, edge_arr, edge_weights_arr)
    graph = Data(x=x_agg, edge_index=edge_index, edge_attr=edge_attr, y=y, names=nodes_arr)
    graph = graph.to(device)
    transform = RandomLinkSplit(num_val=num_val, num_test=num_test, is_undirected=True)
    train_graph, val_graph, test_graph = transform(graph)
    return train_graph, val_graph, test_graph


def train(model,
          train_graph,
          val_graph,
          num_epochs=200,
          batch_size=16,
          lr=0.01,
          lr_step=100,
          lr_gamma=0.98,
          display_freq=100):
    """Trains the model and returns the trained model and the metrics for plotting.
    The metrics are the training and validation loss and accuracy.
    Adam Optimizer is used with a learning rate scheduler.
    """

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    # Lists to store metrics for plotting
    train_loss_values = []
    train_accuracy_values = []
    val_loss_values = []
    val_accuracy_values = []

    for epoch in range(num_epochs):
        model.train()
        train_loader = DataLoader([train_graph], batch_size=batch_size, shuffle=True)
        total_correct_train = 0
        total_samples_train = 0
        total_loss_train = 0

        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()

            # Metrics calculation for training set
            predictions = output.argmax(dim=1)
            correct = predictions.eq(batch.y).sum().item()
            total_correct_train += correct
            total_samples_train += batch.y.size(0)
            total_loss_train += loss.item()

        average_accuracy_train = total_correct_train / total_samples_train
        average_loss_train = total_loss_train / len(train_loader)

        # Validation
        model.eval()
        val_loader = DataLoader([val_graph], batch_size=batch_size, shuffle=False)

        total_correct_val = 0
        total_samples_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for batch in val_loader:
                output = model(batch)
                loss = criterion(output, batch.y)

                # Metrics calculation for validation set
                predictions = output.argmax(dim=1)
                correct = predictions.eq(batch.y).sum().item()
                total_correct_val += correct
                total_samples_val += batch.y.size(0)
                total_loss_val += loss.item()

        average_accuracy_val = total_correct_val / total_samples_val
        average_loss_val = total_loss_val / len(val_loader)

        # Store metrics for plotting
        train_loss_values.append(average_loss_train)
        train_accuracy_values.append(average_accuracy_train)
        val_loss_values.append(average_loss_val)
        val_accuracy_values.append(average_accuracy_val)

        scheduler.step()  # Update learning rate based on the scheduler

        if epoch % (display_freq-1) == 0 and epoch != 0:
            print(f'Epoch {epoch + 1}/{num_epochs} (learning rate: {optimizer.param_groups[0]["lr"]:.3e})')
            print("----------------------------------------------------------")
            print(f"Loss [Train, Val]: [{average_loss_train:.3f}, {average_loss_val:.3f}] \t",
                  f"Accuracy [Train, Val]: [{average_accuracy_train:.3f}, {average_accuracy_val:.3f}]\n")

    model_stats = {"train" : {"loss" : train_loss_values,
                              "accuracy" : train_accuracy_values
                             },
                   "val" : {"loss" : val_loss_values,
                            "accuracy" : val_accuracy_values
                           }
                  }

    return model, model_stats


def plot_results(ax_loss, ax_acc, model_stats):
    """Plots the results of the trained model."""

    # Plotting Loss
    ax_loss.plot(model_stats["train"]["loss"],
                 label="Train",
                 color="blue")
    ax_loss.plot(model_stats["val"]["loss"],
                 "--o",
                 markersize=2.5,
                 label="Validation",
                 color="orange"
                )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross Entropy Loss")
    ax_loss.legend()

    # Plotting Accuracy
    ax_acc.plot(model_stats["train"]["accuracy"],
                label="Train",
                color="blue"
               )
    ax_acc.plot(model_stats["val"]["accuracy"],
                "--o",
                markersize=2.5,
                label="Validation",
                color="orange"
               )
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    return ax_loss, ax_acc
