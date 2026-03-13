"""Tests for eyefeatures/deep/models.py."""

import torch
from torch import nn

from eyefeatures.deep.models import (
    GCN,
    GIN,
    Classifier,
    DSCBlock,
    InceptionBlock,
    Regressor,
    ResnetBlock,
    SimpleRNN,
    VGGBlock,
    VitNet,
    VitNetWithCrossAttention,
    create_simple_CNN,
)


def test_cnn_blocks():
    """Test individual CNN blocks."""
    x = torch.randn(2, 3, 32, 32)

    vgg = VGGBlock(3, 16)
    assert vgg(x).shape == (2, 16, 32, 32)

    resnet = ResnetBlock(3, 3)  # stride=1, same in/out for skip connection
    assert resnet(x).shape == (2, 3, 32, 32)

    inception = InceptionBlock(3, 16, 8, 16, 8, 16, 16)
    # 16 + 16 + 16 + 16 = 64
    assert inception(x).shape == (2, 64, 32, 32)

    dsc = DSCBlock(3, 16)
    assert dsc(x).shape == (2, 16, 32, 32)


def test_create_simple_cnn():
    """Test create_simple_CNN factory."""
    config = {
        0: {"type": "VGG_block", "params": {"out_channels": 16}},
        1: {"type": "Resnet_block", "params": {"out_channels": 16}},
    }
    cnn = create_simple_CNN(config, in_channels=3)
    x = torch.randn(2, 3, 32, 32)
    assert cnn(x).shape == (2, 16, 32, 32)

    # Test Inception block in factory
    config_inc = {
        0: {
            "type": "Inception_block",
            "params": {
                "ch1x1": 8,
                "ch3x3_reduce": 4,
                "ch3x3": 8,
                "ch5x5_reduce": 4,
                "ch5x5": 8,
                "pool_proj": 8,
            },
        }
    }
    cnn_inc = create_simple_CNN(config_inc, in_channels=3)
    assert cnn_inc(x).shape == (2, 32, 32, 32)  # 8*4=32


def test_simple_rnn():
    """Test SimpleRNN."""
    x = torch.randn(2, 10, 5)  # batch, seq, feat
    lengths = torch.tensor([10, 5])

    rnn = SimpleRNN("LSTM", input_size=5, hidden_size=16, bidirectional=True)
    out = rnn(x, lengths)
    assert out.shape == (2, 32)  # 16*2

    # Test return_all
    all_out, hidden = rnn(x, lengths, return_all=True)
    assert all_out.shape == (2, 10, 32)


def test_vit_net():
    """Test VitNet hybrid model."""
    cnn = nn.Sequential(nn.Conv2d(3, 16, 3), nn.AdaptiveAvgPool2d(1))
    embed_dim = 32
    # VitNet projects sequences to embed_dim before RNN
    rnn = SimpleRNN("GRU", input_size=embed_dim, hidden_size=16)

    vit = VitNet(cnn, rnn, fusion_mode="concat", embed_dim=embed_dim)

    images = torch.randn(2, 3, 32, 32)
    sequences = torch.randn(2, 10, 5)  # Input features = 5
    lengths = torch.tensor([10, 8])

    out = vit(images, sequences, lengths)
    # x_1 (32) + x_2 (32) = 64
    assert out.shape == (2, 64)


def test_vit_cross_attention():
    """Test VitNetWithCrossAttention."""
    cnn = nn.Sequential(nn.Conv2d(3, 16, 3), nn.AdaptiveAvgPool2d(1))
    embed_dim = 32
    rnn = SimpleRNN("GRU", input_size=embed_dim, hidden_size=embed_dim)
    # RNN needs padding_idx for mask
    rnn.padding_idx = 0

    vit = VitNetWithCrossAttention(
        cnn, rnn, embed_dim=embed_dim, return_attention_weights=True
    )

    images = torch.randn(2, 3, 32, 32)
    sequences = torch.randn(2, 10, 5)
    lengths = torch.tensor([10, 8])

    out, weights = vit(images, sequences, lengths)
    assert out.shape[0] == 2
    assert weights is not None


def test_classifier():
    """Test Classifier (Lightning module)."""
    backbone = nn.Sequential(nn.Flatten(), nn.LazyLinear(16))
    clf = Classifier(backbone, n_classes=2)

    batch = {"x": torch.randn(4, 32), "y": torch.tensor([0, 1, 0, 1])}  # 32 features

    # Simulate training step which pops y
    batch_copy = batch.copy()
    loss = clf.training_step(batch_copy, 0)
    assert loss > 0

    # Test direct forward with single input dict
    out = clf({"x": torch.randn(4, 32)})
    assert out.shape == (4, 2)


def test_regressor():
    """Test Regressor (Lightning module)."""
    backbone = nn.Sequential(nn.Flatten(), nn.LazyLinear(16))
    reg = Regressor(backbone, output_dim=1)

    batch = {"x": torch.randn(4, 32), "y": torch.randn(4, 1)}

    loss = reg.training_step(batch, 0)
    assert loss >= 0


def test_graph_models():
    """Test GCN and GIN."""

    from torch_geometric.data import Batch, Data

    # Mock some graph data
    x = torch.randn(5, 3)  # 5 nodes, 3 features
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
    )
    edge_attr = torch.randn(6, 1)
    mapping = torch.tensor([0, 1, 2, 3, 4])

    g1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, mapping=mapping)
    batch = Batch.from_data_list([g1, g1])

    gcn = GCN(
        num_nodes=5, feature_dim=3, embedding_dim=8, layer_sizes=[16], out_channels=1
    )
    out = gcn(batch)
    assert out.shape == (2, 16)  # Pooled output

    gin = GIN(
        num_common_nodes=5,
        feature_dim=3,
        embedding_dim=8,
        layer_sizes=[16],
        out_channels=1,
    )
    batch.common_index = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    out_gin = gin(batch)
    assert out_gin.shape == (2, 16)
