from typing import Callable, Dict, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from tqdm import tqdm


class VGGBlock(nn.Module):
    """A VGG-style convolutional block consisting of a
    convolution layer, batch normalization, and ReLU activation.

    Args:
        in_channels: (int) Number of input channels.
        out_channels: (int) Number of output channels.
        kernel_size: (int, optional) Size of the convolution kernel. Default is 3.
        padding: (int, optional) Padding for the convolution layer. Default is 1.
        stride: (int, optional) Stride for the convolution layer. Default is 1.

    Returns:
        Output tensor after applying the convolution, batch normalization, and ReLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ):
        super(VGGBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, images):
        x = self.conv(images)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResnetBlock(nn.Module):
    """A ResNet-style residual block consisting of two convolution layers with skip connections.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolution kernel. Default is 3.
        padding (int, optional): Padding for the convolution layers. Default is 1.
        stride (int, optional): Stride for the convolution layers. Default is 1.

    Returns:
        Output tensor after applying residual connection and ReLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, images):
        identity = images
        out = self.conv1(images)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class InceptionBlock(nn.Module):
    """An Inception-style block consisting of multiple
    convolution branches operating on different scales.

    Args:
        in_channels: (int) Number of input channels.
        ch1x1: (int) Number of output channels for the 1x1 convolution branch.
        ch3x3_reduce: (int) Number of output channels for the 1x1 convolution before the 3x3 convolution.
        ch3x3: (int) Number of output channels for the 3x3 convolution branch.
        ch5x5_reduce: (int) Number of output channels for the 1x1 convolution before the 5x5 convolution.
        ch5x5: (int) Number of output channels for the 5x5 convolution branch.
        pool_proj: (int) Number of output channels for the 1x1 convolution after the max pooling branch.

    Returns:
        Concatenated output tensor from all branches.
    """

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3_reduce: int,
        ch3x3: int,
        ch5x5_reduce: int,
        ch5x5: int,
        pool_proj: int,
    ):
        super(InceptionBlock, self).__init__()

        self.branch1x1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, images):
        branch1x1 = self.branch1x1(images)
        branch3x3 = self.branch3x3(images)
        branch5x5 = self.branch5x5(images)
        branch_pool = self.branch_pool(images)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class DSCBlock(nn.Module):
    """Depthwise separable convolution, which consists of a depthwise
    convolution followed by a pointwise convolution.

    Args:
        in_channels: (int) Number of input channels.
        out_channels: (int) Number of output channels.
        kernel_size: (int, optional) Size of the convolution kernel. Default is 3.
        stride: (int, optional) Stride for the convolution layers. Default is 1.
        padding: (int, optional) Padding for the convolution layers. Default is 1.

    Returns:
        Output tensor after applying depthwise and pointwise convolutions, batch normalization, and ReLU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DSCBlock, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, images):
        x = self.depthwise(images)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


_blocks = {
    "VGG_block": VGGBlock,
    "Resnet_block": ResnetBlock,
    "Inception_block": InceptionBlock,
    "DSC_block": DSCBlock,
}


def create_simple_CNN(
    config: Dict[int, Dict[str, Union[str, Dict]]],
    in_channels: int,
    shape: Tuple[int, int] = None,
):
    """Creates a simple CNN based on the provided configuration.

    Args:
        config: (Dict) Configuration dictionary where each key represents a layer/block and its corresponding parameters.
        in_channels: (int) Number of input channels for the first layer.
        shape: (Tuple[int, int], optional) Input shape for the CNN. If provided, checks that the final output shape is valid.

    Returns: (nn.Sequential, Tuple[int, int]) or nn.Sequential
        cnn: (nn.Sequential)
            Sequential CNN model.
        shape: (Tuple[int, int], optional)
            Final output shape, if provided.
    """

    modules = list()

    for idx, block_config in tqdm(config.items()):
        block_type = block_config["type"]
        params = block_config.get("params", {})

        if block_type != "Inception_block":
            module = _blocks[block_type](in_channels=in_channels, **params)
            in_channels = params.get("out_channels", in_channels)
            if shape is not None:
                kernel_size = params.get("kernel_size", 3)
                padding = params.get("padding", 1)
                stride = params.get("stride", 1)
                shape = (
                    (shape[0] - kernel_size + 2 * padding) // stride + 1,
                    (shape[1] - kernel_size + 2 * padding) // stride + 1,
                )
                if shape[0] < 1 or shape[1] < 1:
                    raise ValueError(
                        """Your CNN backbone is too large for the input shape! 
                    Increase resolution or consider reducing the number of layers/removing stride/adding padding."""
                    )
        else:
            module = _blocks[block_type](in_channels=in_channels, **params)
            in_channels = sum(
                [params.get(key) for key in ["ch1x1", "ch3x3", "ch5x5", "pool_proj"]]
            )

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
        num_layers=1,
        bidirectional=False,
        pre_rnn_linear_size=None,
    ):
        """A simple recurrent neural network (RNN) module
        that supports RNN, LSTM, and GRU architectures.

        Args:
            rnn_type: (str) Type of RNN ('RNN', 'LSTM', or 'GRU').
            input_size: (int) Number of input features.
            hidden_size: (int) Number of hidden units.
            num_layers: (int, optional) Number of RNN layers. Default is 1.
            bidirectional: (bool, optional) Whether the RNN is bidirectional. Default is False.
            pre_rnn_linear_size: (int, optional) Size of the optional linear layer before the RNN.

        Returns:
            Output tensor after the RNN and fully connected layer.
        """
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
        if rnn_type == "RNN":
            self.rnn = nn.RNN(
                self.input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                self.input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                self.input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(
                f"Unsupported rnn_type: {rnn_type}. Choose from 'RNN', 'LSTM', 'GRU'."
            )

    def forward(self, sequences, lengths):
        # Apply the optional linear layer before the RNN
        x = sequences
        if self.pre_rnn_linear is not None:
            x = self.pre_rnn_linear(x)

        # Pack the padded sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # Forward pass through the RNN
        packed_out, hidden = self.rnn(packed_x)

        # Unpack the sequence
        # out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # If the RNN is bidirectional, concatenate the hidden states from both directions
        if self.bidirectional:
            if isinstance(
                hidden, tuple
            ):  # For LSTM, hidden is a tuple (hidden_state, cell_state)
                hidden = torch.cat((hidden[0][-2], hidden[0][-1]), dim=1)
            else:  # For RNN and GRU
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            if isinstance(hidden, tuple):  # For LSTM
                hidden = hidden[0][-1]
            else:  # For RNN and GRU
                hidden = hidden[-1]
        # Pass the last hidden state through the fully connected layer
        return hidden


class VitNet(nn.Module):
    """Parent class for a vision-and-text network that
    fuses CNN and RNN-based representations using
    concatenation or addition.

    Args:
        CNN: (nn.Module) CNN backbone for processing image data.
        RNN: (nn.Module) RNN backbone for processing sequence data.
        fusion_mode: (str, optional) Fusion mode ('concat' or 'add'). Default is 'concat'.
        activation: (nn.Module, optional) Activation function applied after fusion. Default is None.
        embed_dim: (int, optional) Embedding dimension for the projected features. Default is 128.
    """

    def __init__(self, CNN, RNN, fusion_mode="concat", activation=None, embed_dim=32):
        super(VitNet, self).__init__()
        self.CNN = CNN
        self.RNN = RNN
        self.fusion_mode = fusion_mode
        self.activation = activation
        self.image_proj = nn.LazyLinear(embed_dim)
        self.rnn_proj = nn.LazyLinear(embed_dim)
        self.flat = nn.Flatten()

    def forward(self, images, sequences, lengths):
        x_1 = self.image_proj(self.flat(self.CNN(images)))

        projected_sequences = self.rnn_projection(sequences)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            projected_sequences, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, x_2 = self.RNN(packed_input)
        _, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, padding_value=0
        )

        if self.fusion_mode == "add":
            x = x_1 + x_2
        elif self.fusion_mode == "concat":
            x = torch.cat([x_1, x_2], dim=1)
        else:
            raise NotImplementedError(
                f"Fusion mode {self.fusion_mode} is not implemented."
            )

        if self.activation is not None:
            x = self.activation(x)

        return x


class VitNetWithCrossAttention(VitNet):
    """Child class extending VitNet by adding cross-attention
    between image and sequence representations.

    Args:
        CNN: (nn.Module) CNN backbone for processing image data.
        RNN: (nn.Module) RNN backbone for processing sequence data.
        input_dim: (int) Input dimension for the RNN.
        projected_dim: (int) Dimension of the projected sequence representation.
        cross_attention_fusion_mode: (str, optional) Fusion mode after cross-attention ('concat' or 'add'). Default is 'concat'.
        activation: (nn.Module, optional) Activation function applied after fusion. Default is None.
        embed_dim: (int, optional) Embedding dimension for the projected features. Default is 128.
        return_attention_weights: (bool, optional) Whether to return attention weights. Default is False.
    """

    def __init__(
        self,
        CNN,
        RNN,
        input_dim: int,
        projected_dim: int,
        cross_attention_fusion_mode="concat",
        activation=None,
        embed_dim=128,
        return_attention_weights=False,
    ):
        super(VitNetWithCrossAttention, self).__init__(
            CNN,
            RNN,
            input_dim,
            projected_dim,
            fusion_mode="concat",
            activation=activation,
            embed_dim=embed_dim,
        )
        self.cross_attention_layer = nn.MultiheadAttention(embed_dim, num_heads=8)
        self.cross_attention_fusion_mode = cross_attention_fusion_mode
        self.return_attention_weights = return_attention_weights
        self.padding_idx = RNN.padding_idx

    def forward(self, images, sequences, lengths):
        # First, run the parent class's forward method to get the basic fusion output
        x = super().forward(images, sequences, lengths)

        # Cross-attention mechanism
        x_1 = self.image_proj(self.CNN(images))
        projected_sequences = self.sequence_projection(sequences)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            projected_sequences, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.RNN(packed_input)
        all_hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, padding_value=0
        )

        query = self.image_proj(x_1).unsqueeze(0)  # (1, batch_size, embed_dim)
        key = self.rnn_proj(all_hidden).transpose(0, 1)
        key_padding_mask = sequences[:, :, 0] == self.padding_idx
        context_vector, attention_weights = self.cross_attention_layer(
            query, key, key, key_padding_mask=key_padding_mask
        )

        output = context_vector.squeeze(0)
        if self.cross_attention_fusion_mode == "concat":
            x = torch.cat([x, output], dim=1)
        else:
            x = output

        if self.activation is not None:
            x = self.activation(x)

        if self.return_attention_weights:
            return x, attention_weights

        return x


class GCN(torch.nn.Module):
    """A graph convolutional network (GCN)
    with optional learnable node embeddings.

    Args:
        num_nodes: (int) Number of nodes in the graph.
        feature_dim: (int) Dimensionality of node features.
        embedding_dim: (int) Dimensionality of the learnable node embeddings.
        layer_sizes: (List[int]) List of hidden layer sizes for each GCN layer.
        out_channels: (int) Number of output channels.
        use_embeddings: (bool, optional) Whether to use learnable embeddings. Default is True.

    Returns:
        Output tensor after graph convolutions and global mean pooling.
    """

    def __init__(
        self,
        num_nodes,
        feature_dim,
        embedding_dim,
        layer_sizes,
        out_channels,
        use_embeddings=True,
    ):
        super(GCN, self).__init__()
        self.use_embeddings = use_embeddings
        if use_embeddings:
            self.embeddings = torch.nn.Embedding(
                num_nodes, embedding_dim
            )  # Learnable embeddings

        in_channels = feature_dim + (embedding_dim if use_embeddings else 0)
        self.convs = torch.nn.ModuleList()  # List of GCN layers
        for hidden_channels in layer_sizes:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            in_channels = hidden_channels  # Update in_channels for next layer

    def forward(self, graphs):
        if self.use_embeddings:
            # Combine node features and learnable embeddings
            node_embeddings = self.embeddings.weight[
                graphs.mapping
            ]  # Retrieve embeddings for each node
            x = torch.cat([graphs.x, node_embeddings], dim=1)
        else:
            # Use only node features
            x = graphs.x

        # Apply each GCN layer with optional edge weights
        for conv in self.convs:
            x = conv(x, graphs.edge_index, edge_weight=graphs.edge_attr)
            x = F.relu(x)

        # Global pooling (mean pooling)
        x = global_mean_pool(x, graphs.batch)

        return x


class GIN(torch.nn.Module):
    """A graph isomorphism network (GIN) with shared
    embeddings for nodes across graphs.

    Args:
        num_common_nodes: (int) Number of shared nodes across graphs.
        feature_dim: (int) Dimensionality of node features.
        embedding_dim: (int) Dimensionality of the shared node embeddings.
        layer_sizes: (List[int]) List of hidden layer sizes for each GIN layer.
        out_channels: (int) Number of output channels.

    Returns:
        Output tensor after GIN layers and global mean pooling.
    """

    def __init__(
        self, num_common_nodes, feature_dim, embedding_dim, layer_sizes, out_channels
    ):
        super(GIN, self).__init__()
        # Shared embeddings across graphs
        self.embeddings = torch.nn.Embedding(num_common_nodes, embedding_dim)

        # Determine the input dimension for the first GIN layer
        in_channels = feature_dim + embedding_dim

        self.convs = torch.nn.ModuleList()  # List of GIN layers
        for hidden_channels in layer_sizes:
            nn = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(nn))
            in_channels = hidden_channels  # Update in_channels for next layer

    def forward(self, graphs):
        # Retrieve the shared embeddings using the common index
        shared_embeddings = self.embeddings(graphs.common_index)

        # Concatenate node features with the shared embeddings
        x = torch.cat([graphs.x, shared_embeddings], dim=1)

        # Apply each GIN layer
        for conv in self.convs:
            x = conv(x, graphs.edge_index)
            x = F.relu(x)

        # Global pooling (mean pooling)
        x = global_mean_pool(x, graphs.batch)

        return x


class BaseModel(pl.LightningModule):
    """A base PyTorch Lightning module for neural networks
    with optional custom optimizer and scheduler.

    Args:
        backbone: (Union[nn.ModuleList, nn.Module]) The feature extraction backbone model.
        output_size: (int) Size of the final output layer.
        hidden_layers: (Tuple, optional) Tuple of hidden layer sizes. Default is empty.
        activation: (nn.Module, optional) Activation function to apply between layers. Default is ReLU.
        learning_rate: (float, optional) Learning rate for the optimizer. Default is 1e-3.
        optimizer_class: (Callable, optional) Optimizer class to use. Default is AdamW.
        optimizer_params: (dict, optional) Additional parameters for the optimizer. Default is None.
        scheduler_class: (Callable, optional) Scheduler class to use. Default is None.
        scheduler_params: (dict, optional) Additional parameters for the scheduler. Default is None.
        loss_fn: (Callable, optional) Loss function to use. Default is None.

    Returns:
        Output after the forward pass through the network.
    """

    def __init__(
        self,
        backbone: Union[nn.ModuleList, nn.Module],
        output_size,
        hidden_layers: Tuple = (),
        activation=nn.ReLU(),
        learning_rate: float = 1e-3,
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

        self.flat = nn.Flatten()

    def forward(self, x):
        if isinstance(self.backbone, nn.Sequential):
            x = self.backbone(*x.values())
        else:
            x = self.backbone(**x)
        if len(x.size()) == 4:
            x = self.flat(x)
        for layer in self.head:
            x = layer(x)
        return x

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(), lr=self.LR, **self.optimizer_params
        )

        if self.scheduler_class is not None:
            scheduler = self.scheduler_class(optimizer, **self.scheduler_params)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        y = batch.pop("y")
        out = self(batch)
        loss = self.loss_fn(out, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.pop("y")
        out = self(batch)
        loss = self.loss_fn(out, y)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchmetrics


class Classifier(BaseModel):
    """A classification model built on top of the BaseModel,
    with additional accuracy, precision, recall, and F1-score tracking.

    Args:
        backbone: (Union[nn.ModuleList, nn.Module]) The feature extraction backbone model.
        n_classes: (int) Number of output classes.
        classifier_hidden_layers: (Tuple, optional) Tuple of hidden layer sizes for the classifier. Default is empty.
        classifier_activation: (nn.Module, optional) Activation function to use in the classifier. Default is ReLU.
        learning_rate: (float, optional) Learning rate for the optimizer. Default is 1e-3.
        optimizer_class: (Callable, optional) Optimizer class to use. Default is AdamW.
        optimizer_params: (dict, optional) Additional parameters for the optimizer. Default is None.
        scheduler_class: (Callable, optional) Scheduler class to use. Default is None.
        scheduler_params: (dict, optional) Additional parameters for the scheduler. Default is None.

    Returns:
        Output logits after the forward pass through the classifier.
    """

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
            loss_fn=nn.CrossEntropyLoss(),
        )

        self.accuracy = torchmetrics.Accuracy(
            task="binary" if n_classes == 2 else "multiclass", num_classes=n_classes
        )

        self.precision = torchmetrics.Precision(
            task="binary" if n_classes == 2 else "multiclass",
            num_classes=n_classes,
            average=None,  # Per-class precision
        )

        self.recall = torchmetrics.Recall(
            task="binary" if n_classes == 2 else "multiclass",
            num_classes=n_classes,
            average=None,  # Per-class recall
        )

        self.f1 = torchmetrics.F1Score(
            task="binary" if n_classes == 2 else "multiclass",
            num_classes=n_classes,
            average=None,  # Per-class F1-score
        )

        # For macro average metrics
        self.macro_precision = torchmetrics.Precision(
            task="binary" if n_classes == 2 else "multiclass",
            num_classes=n_classes,
            average="macro",  # Macro-average precision
        )

        self.macro_recall = torchmetrics.Recall(
            task="binary" if n_classes == 2 else "multiclass",
            num_classes=n_classes,
            average="macro",  # Macro-average recall
        )

        self.macro_f1 = torchmetrics.F1Score(
            task="binary" if n_classes == 2 else "multiclass",
            num_classes=n_classes,
            average="macro",  # Macro-average F1-score
        )

        self.prob = nn.Softmax(dim=1)

    def validation_step(self, batch, batch_idx):
        y = batch.pop("y")  # Assuming that "y" is the ground truth labels
        out = self(batch)  # Forward pass
        loss = self.loss_fn(out, y)
        out = self.prob(out)
        logits = torch.argmax(out, dim=1)

        # Calculate metrics
        accu = self.accuracy(logits, y)
        precision = self.precision(logits, y)
        recall = self.recall(logits, y)
        f1 = self.f1(logits, y)

        macro_precision = self.macro_precision(logits, y)
        macro_recall = self.macro_recall(logits, y)
        macro_f1 = self.macro_f1(logits, y)

        # Log loss and accuracy
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc_step", accu, on_step=False, on_epoch=True, prog_bar=False)

        # Log precision, recall, and F1-score for each class
        for i in range(len(precision)):
            self.log(
                f"val_precision_class_{i}",
                precision[i],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                f"val_recall_class_{i}",
                recall[i],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                f"val_f1_class_{i}", f1[i], on_step=False, on_epoch=True, prog_bar=False
            )

        # Log macro averages
        self.log(
            "val_macro_precision",
            macro_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val_macro_recall",
            macro_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("val_macro_f1", macro_f1, on_step=False, on_epoch=True, prog_bar=False)


class Regressor(BaseModel):
    """A regression model built on top of the BaseModel,
    using mean squared error loss.

    Args:
        backbone: (Union[nn.ModuleList, nn.Module]) The feature extraction backbone model.
        output_dim: (int) Dimensionality of the regression output.
        regressor_hidden_layers: (Tuple, optional) Tuple of hidden layer sizes for the regressor. Default is empty.
        regressor_activation: (nn.Module, optional) Activation function to use in the regressor. Default is ReLU.
        learning_rate: (float, optional) Learning rate for the optimizer. Default is 1e-3.
        optimizer_class: (Callable, optional) Optimizer class to use. Default is AdamW.
        optimizer_params: (dict, optional) Additional parameters for the optimizer. Default is None.
        scheduler_class: (Callable, optional) Scheduler class to use. Default is None.
        scheduler_params: (dict, optional) Additional parameters for the scheduler. Default is None.

    Returns:
        Regression output after the forward pass.
    """

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
            loss_fn=nn.MSELoss(),
        )
