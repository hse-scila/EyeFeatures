Models
======

The ``models`` submodule implements common neural network architectures in ``pytorch``
to work with eyetracking data. There are two main classes - Classifier and Regressor,
to solve classification and regression tasks, respectively. They are wrappers
upon the ``backbone`` parameter, which could be built using ``pytorch``-implemented
blocks.

Building Blocks
***************

The blocks implemented in ``models`` to construct the ``backbone``:

.. toctree::
    :maxdepth: 1

    vgg_block
    resnet_block
    inception_block
    dsc_block

Backbones
*********

.. toctree::
    :maxdepth: 1

    simple_rnn
    vit_net
    vit_net_with_cross_attention
    gcn
    gin

Models
******

.. toctree::
    :maxdepth: 1

    classifier
    regressor

Functions
*********

Methods that combine above classes and wrappers.

.. toctree::
    :maxdepth: 1

    create_simple_CNN
