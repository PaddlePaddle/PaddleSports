import paddle
import paddle.nn as nn
from paddle.nn import Linear, Conv2D, BatchNorm1D, BatchNorm2D, PReLU, ReLU, Sigmoid, Dropout, MaxPool2D, \
    AdaptiveAvgPool2D, Sequential, Layer
from collections import namedtuple


# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']


class Flatten(Layer):
    def forward(self, x):
        return paddle.reshape(x,(x.shape[0],-1))


def l2_norm(x, axis=1):
    norm = paddle.norm(input, 2, axis, True)
    output = paddle.divide(input, norm)

    return output


class SEModule(Layer):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        self.fc1 = Conv2D(
            channels, channels // reduction, kernel_size=1, padding=0,weight_attr=weight_attr)

        self.relu = ReLU()
        self.fc2 = Conv2D(
            channels // reduction, channels, kernel_size=1, padding=0)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Layer):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2D(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2D(in_channel, depth, (1, 1), stride), BatchNorm2D(depth))
        self.res_layer = Sequential(
            BatchNorm2D(in_channel),
            Conv2D(in_channel, depth, (3, 3), (1, 1), 1), PReLU(depth),
            Conv2D(depth, depth, (3, 3), stride, 1), BatchNorm2D(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Layer):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2D(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2D(in_channel, depth, (1, 1), stride),
                BatchNorm2D(depth))
        self.res_layer = Sequential(
            BatchNorm2D(in_channel),
            Conv2D(in_channel, depth, (3, 3), (1, 1), 1),
            PReLU(depth),
            Conv2D(depth, depth, (3, 3), stride, 1),
            BatchNorm2D(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


class Backbone(Layer):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2D(3, 64, (3, 3), 1, 1),
                                      BatchNorm2D(64),
                                      PReLU(64))
        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2D(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1D(512))
        else:
            self.output_layer = Sequential(BatchNorm2D(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1D(512))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
                m.weight_attr = weight_attr
            elif isinstance(m, nn.BatchNorm2D):
                weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
                m.weight_attr = weight_attr
            elif isinstance(m, nn.BatchNorm1D):
                weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(1.0))
                m.weight_attr = weight_attr
            elif isinstance(m, nn.Linear):
                weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
                m.weight_attr = weight_attr


def IR_50(input_size):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')

    return model


def IR_101(input_size):
    """Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir')

    return model


def IR_152(input_size):
    """Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')

    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')

    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')

    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')

    return model
