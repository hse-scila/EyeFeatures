import pytorch_lightning as pl
import torch
import torchmetrics
from sklearn.model_selection import train_test_split
from torch import nn
from typing import Union, List, Tuple, Dict
from tqdm import tqdm
import pandas as pd
from scipy.stats import gaussian_kde
from numpy.typing import NDArray
import numpy as np
import warnings
from torch.nn import functional as F
import torchmetrics
import pytorch_lightning as pl
from typing import Union, Callable, Optional
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool
from eyetracking.features.complex import get_heatmaps
#from utils import get_heatmaps


class VGGBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3,
                 padding: int = 1,
                 stride: int = 1):
        super(VGGBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class ResnetBlock(nn.Module):
    def __init__(self,
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3,
                 padding: int = 1,
                 stride: int = 1):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        
        return out
    
class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, 
                 ch1x1: int, 
                 ch3x3_reduce: int, 
                 ch3x3: int, 
                 ch5x5_reduce: int, 
                 ch5x5: int, 
                 pool_proj: int):
        
        super(InceptionBlock, self).__init__()
        
        self.branch1x1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)
    

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
_blocks = {'VGG_block': VGGBlock,
           'Resnet_block': ResnetBlock,
           'Inception_block': InceptionBlock,
           'DepthwiseConvBlock': DepthwiseSeparableConv}
    

def create_simple_CNN(
        config: Dict[int, Dict[str, Union[str, Dict]]],
        in_channels: int,
        shape: Tuple[int, int] = None):
    
    modules = list()

    for idx, block_config in tqdm(config.items()):
        block_type = block_config['type']
        params = block_config.get('params', {})
        
        if block_type != 'Inception_block': 
            module = _blocks[block_type](
                in_channels=in_channels,
                **params
            )
            in_channels = params.get('out_channels', in_channels)
            if shape is not None:
                kernel_size = params.get('kernel_size', 3)
                padding = params.get('padding', 1)
                stride = params.get('stride', 1)
                shape = (
                    (shape[0] - kernel_size + 2*padding) // stride + 1,
                    (shape[1] - kernel_size + 2*padding) // stride + 1,
                )
                if shape[0] < 1 or shape[1] < 1:
                    raise ValueError('''Your CNN backbone is too large for the input shape! 
                    Increase resolution or consider reducing the number of layers/removing stride/adding padding.''')
        else:

            module = _blocks[block_type](
                in_channels=in_channels,
                **params
            )
            in_channels = sum([params.get(key) for key in ['ch1x1', 'ch3x3', 'ch5x5', 'pool_proj']])

        modules.append(module)

    cnn = nn.Sequential(*modules)
    
    if shape is not None:
        return cnn, shape
    return cnn


class SimpleRNN(nn.Module):
    def __init__(
        self, 
        rnn_type, 
        input_size, 
        hidden_size, 
        output_size, 
        num_layers=1, 
        bidirectional=False, 
        pre_rnn_linear_size=None
    ):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Optional linear layer before the RNN
        if pre_rnn_linear_size is not None:
            self.pre_rnn_linear = nn.Linear(input_size, pre_rnn_linear_size)
            self.input_size = pre_rnn_linear_size  # Update input_size for the RNN layer
        else:
            self.pre_rnn_linear = None
            self.input_size = input_size  # Original input size

        # Set the RNN layer based on the rnn_type parameter
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(self.input_size, hidden_size, num_layers, 
                              batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, hidden_size, num_layers, 
                               batch_first=True, bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(self.input_size, hidden_size, num_layers, 
                              batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Choose from 'RNN', 'LSTM', 'GRU'.")
        
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
    
    def forward(self, x, lengths):
        # Apply the optional linear layer before the RNN
        if self.pre_rnn_linear is not None:
            x = self.pre_rnn_linear(x)

        # Pack the padded sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, 
                                                     batch_first=True, enforce_sorted=False)
        
        # Forward pass through the RNN
        packed_out, hidden = self.rnn(packed_x)
        
        # Unpack the sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # If the RNN is bidirectional, concatenate the hidden states from both directions
        if self.bidirectional:
            if isinstance(hidden, tuple):  # For LSTM, hidden is a tuple (hidden_state, cell_state)
                hidden = torch.cat((hidden[0][-2], hidden[0][-1]), dim=1)
            else:  # For RNN and GRU
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            if isinstance(hidden, tuple):  # For LSTM
                hidden = hidden[0][-1]
            else:  # For RNN and GRU
                hidden = hidden[-1]
        
        # Pass the last hidden state through the fully connected layer
        output = self.fc(hidden)
        return output


class VitNet(nn.Module):
    def __init__(self, 
                 CNN, 
                 RNN, 
                 input_dim: int, 
                 projected_dim: int,  # dimensionality of the projected sequence representation
                 fusion_mode = 'concat',
                 activation = None, 
                 cross_attention = False, 
                 cross_attention_fusion_mode = 'concat',
                 embed_dim = 128,
                 return_attention_weights = False,
                 ):
        super(VitNet, self).__init__()
        self.sequence_projection = nn.Linear(input_dim, projected_dim)
        self.CNN = CNN
        self.RNN = RNN
        self.fusion_mode = fusion_mode
        self.activation = activation
        self.image_proj = nn.Linear(self.CNN.output_dim, embed_dim)
        self.rnn_proj = nn.Linear(self.RNN.hidden_dim, embed_dim)
        self.cross_attention = cross_attention
        if self.cross_attention:
            self.padding_idx = self.RNN.padding_idx
            self.cross_attention_layer = nn.MultiheadAttention(self.RNN.hidden_size)
            self.cross_attention_fusion_mode = cross_attention_fusion_mode
            self.return_attention_weights = return_attention_weights
            
    
    def forward(self, images, sequences, sequences_length):
        
        x_1 = self.image_proj(self.CNN(images))

        projected_sequences = self.input_projection(sequences)
        packed_input = nn.utils.rnn.pack_padded_sequence(projected_sequences, sequences_length, batch_first=True, enforce_sorted=False)
        packed_output, x_2 = self.rnn(packed_input)
        all_hidden, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, padding_value=0)
        
        if self.fusion_mode == 'add':
            x = x_1 + x_2
        elif self.fusion_mode == 'concat':
            x = torch.cat([x_1, x_2], axis=1)
        elif self.fusion_mode == 'attention_concat':
            x = x_1

        if self.activation is not None:
            x = self.activation(x)

        if self.cross_attention != False:
            query = self.image_proj(x_1).unsqueeze(0)  # (1, batch_size, embed_dim)
            key = self.rnn_proj(all_hidden).transpose(0, 1)
            key_padding_mask = (sequences[:, :, 0] == self.padding_idx)
            context_vector, attention_weights = self.cross_attention(query, key, key, key_padding_mask=key_padding_mask)
            output = self.fc(context_vector.squeeze(0))
            if self.cross_attention_fusion_mode == 'concat':
                x = torch.cat([x, output], dim=1)
            else:
                x = output
            if self.return_attention_weights:
                return x, attention_weights
        return x
    

class GCN(torch.nn.Module):
    def __init__(self, num_nodes, feature_dim, embedding_dim, layer_sizes, out_channels, use_embeddings=True):
        super(GCN, self).__init__()
        self.use_embeddings = use_embeddings
        if use_embeddings:
            self.embeddings = torch.nn.Embedding(num_nodes, embedding_dim)  # Learnable embeddings

        in_channels = feature_dim + (embedding_dim if use_embeddings else 0)
        self.convs = torch.nn.ModuleList()  # List of GCN layers
        for hidden_channels in layer_sizes:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            in_channels = hidden_channels  # Update in_channels for next layer


    def forward(self, data):
        if self.use_embeddings:
            # Combine node features and learnable embeddings
            node_embeddings = self.embeddings.weight[data.mapping]  # Retrieve embeddings for each node
            x = torch.cat([data.x, node_embeddings], dim=1)
        else:
            # Use only node features
            x = data.x

        # Apply each GCN layer with optional edge weights
        for conv in self.convs:
            x = conv(x, data.edge_index, edge_weight=data.edge_attr)
            x = F.relu(x)

        # Global pooling (mean pooling)
        x = global_mean_pool(x, data.batch)
        
        
        return x 
    

class GIN(torch.nn.Module):
    def __init__(self, num_common_nodes, feature_dim, embedding_dim, layer_sizes, out_channels):
        super(GIN, self).__init__()
        # Shared embeddings across graphs
        self.embeddings = torch.nn.Embedding(num_common_nodes, embedding_dim)

        # Determine the input dimension for the first GIN layer
        in_channels = feature_dim + embedding_dim
        
        self.convs = torch.nn.ModuleList()  # List of GIN layers
        for hidden_channels in layer_sizes:
            nn = nn.Sequential(nn.Linear(in_channels, hidden_channels), 
                               nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
            self.convs.append(GINConv(nn))
            in_channels = hidden_channels  # Update in_channels for next layer


    def forward(self, data):
        # Retrieve the shared embeddings using the common index
        shared_embeddings = self.embeddings(data.common_index)

        # Concatenate node features with the shared embeddings
        x = torch.cat([data.x, shared_embeddings], dim=1)

        # Apply each GIN layer
        for conv in self.convs:
            x = conv(x, data.edge_index)
            x = F.relu(x)

        # Global pooling (mean pooling)
        x = global_mean_pool(x, data.batch)
        
        return x

class BaseModel(pl.LightningModule):
    def __init__(
        self,
        backbone: Union[nn.ModuleList, nn.Module],
        output_size,
        hidden_layers=(),
        activation=nn.ReLU(),
        learning_rate=1e-3,
        optimizer_class: Callable = torch.optim.AdamW,
        optimizer_params: Optional[dict] = None,
        scheduler_class: Optional[Callable] = None,
        scheduler_params: Optional[dict] = None,
        loss_fn: Optional[Callable] = None,
    ):
        super().__init__()

        self.backbone = backbone

        modules = list()

        for hidden_units in hidden_layers:
            modules.append(nn.LazyLinear(hidden_units))
            modules.append(activation)
            in_features = hidden_units
        modules.append(nn.LazyLinear(output_size))
        self.head = nn.ModuleList(modules)

        self.loss_fn = loss_fn
        self.LR = learning_rate

        # Optimizer and Scheduler configuration
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params if scheduler_params is not None else {}

    def forward(self, x):
        x = self.backbone(x)
        x = self.flat(x)
        for layer in self.head:
            x = layer(x)
        return x

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.LR, **self.optimizer_params)

        if self.scheduler_class is not None:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_params)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        out = self(x)
        loss = self.loss_fn(out, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        out = self(x)
        loss = self.loss_fn(out, y)
        self.log("valid_loss", loss)
        return loss
    

class Classifier(BaseModel):
    def __init__(
        self,
        backbone: Union[nn.ModuleList, nn.Module],
        n_classes,
        classifier_hidden_layers=(),
        classifier_activation=nn.ReLU(),
        learning_rate=1e-3,
        optimizer_class: Callable = torch.optim.AdamW,
        optimizer_params: Optional[dict] = None,
        scheduler_class: Optional[Callable] = None,
        scheduler_params: Optional[dict] = None,
    ):
        super().__init__(
            backbone=backbone,
            output_size=n_classes,
            hidden_layers=classifier_hidden_layers,
            activation=classifier_activation,
            learning_rate=learning_rate,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
            loss_fn=nn.CrossEntropyLoss()
        )

        self.accuracy = torchmetrics.Accuracy(
            task="binary" if n_classes == 2 else "multiclass"
        )
        self.prob = nn.Softmax(dim=1)

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        out = self(x)
        loss = self.loss_fn(out, y)
        out = self.prob(out)
        logits = torch.argmax(out, dim=1)
        accu = self.accuracy(logits, y)
        self.log("valid_loss", loss)
        self.log("val_acc_step", accu)


class Regressor(BaseModel):
    def __init__(
        self,
        backbone: Union[nn.ModuleList, nn.Module],
        output_dim,
        regressor_hidden_layers=(),
        regressor_activation=nn.ReLU(),
        learning_rate=1e-3,
        optimizer_class: Callable = torch.optim.AdamW,
        optimizer_params: Optional[dict] = None,
        scheduler_class: Optional[Callable] = None,
        scheduler_params: Optional[dict] = None,
    ):
        super().__init__(
            backbone=backbone,
            output_size=output_dim,
            hidden_layers=regressor_hidden_layers,
            activation=regressor_activation,
            learning_rate=learning_rate,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            scheduler_class=scheduler_class,
            scheduler_params=scheduler_params,
            loss_fn=nn.MSELoss()
        )
