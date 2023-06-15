import torch
from torch.nn.functional import ctc_loss, log_softmax
from torch.nn import Sequential, GRU, Module, AvgPool2d, Conv2d, Linear, LSTM
from torchvision import models


class FeatureExtractor(Module):
    def __init__(self, input_size, output_len):
        super(self.__class__, self).__init__()

        h, w = input_size
        resnet = getattr(models, "resnet18")(weights=True)
        self.cnn = Sequential(*list(resnet.children())[:-2])

        self.pool = AvgPool2d(kernel_size=(h // 32, 1))
        self.proj = Conv2d(w // 32, output_len, kernel_size=1)

        self.num_output_features = self.cnn[-1][-1].bn2.num_features

    # class FeatureExtractor(Module):
    #     def __init__(self, input_size, output_len):
    #         super(self.__class__, self).__init__()
    #
    #         h, w = input_size
    #         resnet = getattr(models, "efficientnet_b4")(weights=True)
    #         self.cnn = Sequential(*list(resnet.children())[:-2])
    #
    #         self.pool = AvgPool2d(kernel_size=(h // 32, 1))
    #         self.proj = Conv2d(w // 32, output_len, kernel_size=1)
    #
    #         # self.num_output_features = self.cnn[-1][-1].bn2.num_features
    #         self.num_output_features = self.cnn[-1][-1][-2].num_features

    def apply_projection(self, x):
        """Use convolution to increase width of a features.

        Args:
            - x: Tensor of features (shaped B x C x H x W).

        Returns:
            New tensor of features (shaped B x C x H x W').
        """
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        return x

    def forward(self, x):
        # print(x.size())
        # Apply conv layers
        features = self.cnn(x)

        # Pool to make height == 1
        features = self.pool(features)

        # Apply projection to increase width
        features = self.apply_projection(features)

        return features


class SequencePredictor(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout,
        bidirectional,
    ):
        super(self.__class__, self).__init__()

        self.num_classes = num_classes
        self.rnn = GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        fc_in = hidden_size if not bidirectional else 2 * hidden_size
        self.fc = Linear(in_features=fc_in, out_features=num_classes)

    def _init_hidden(self, batch_size):
        """Initialize new tensor of zeroes for RNN hidden state.

        Args:
            - batch_size: Int size of batch

        Returns:
            Tensor of zeros shaped (num_layers * num_directions, batch, hidden_size).
        """
        num_directions = 2 if self.rnn.bidirectional else 1

        h = torch.zeros(
            self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size
        )

        return h

    @staticmethod
    def _reshape_features(x):
        """Change dimensions of x to fit RNN expected input.

        Args:
            - x: Tensor x shaped (B x (C=1) x H x W).

        Returns:
            New tensor shaped (W x B x H).
        """

        x = x.squeeze(1)
        x = x.permute(2, 0, 1)

        return x

    def forward(self, x):
        x = self._reshape_features(x)

        batch_size = x.size(1)
        h_0 = self._init_hidden(batch_size)
        h_0 = h_0.to(x.device)
        x, h = self.rnn(x, h_0)

        x = self.fc(x)
        return x


class CRNN(Module):
    def __init__(
        self,
        num_chars,
        cnn_input_size,
        cnn_output_len,
        rnn_hidden_size,
        rnn_num_layers,
        rnn_dropout,
        rnn_bidirectional,
    ):
        super(self.__class__, self).__init__()

        self.num_chars = num_chars + 1
        self.features_extractor = FeatureExtractor(
            input_size=cnn_input_size, output_len=cnn_output_len
        )
        self.sequence_predictor = SequencePredictor(
            input_size=self.features_extractor.num_output_features,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            num_classes=self.num_chars,
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional,
        )

    def forward(self, images, targets=None, seq_len=None):
        batch_size, _, _, _ = images.size()

        x = self.features_extractor(images)
        x = self.sequence_predictor(x)

        if targets is not None:
            log_softmax_values = log_softmax(x, dim=2)
            # print(f"{log_softmax_values.shape=}")
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=log_softmax_values.size(0),
                dtype=torch.int32,
            )
            # # print(input_lengths)
            # target_lengths = torch.full(
            #     size=(batch_size,), fill_value=targets.size(1), dtype=torch.int32
            # )

            loss = ctc_loss(
                log_probs=log_softmax_values.cpu(),  # (T, N, C)
                targets=targets,  # N, S or sum(target_lengths)
                input_lengths=input_lengths,  # N
                target_lengths=seq_len,
                zero_infinity=True,
            )  # N
            # print(target_lengths)
            # loss = nn.CTCLoss(blank=0)(
            #     log_softmax_values, targets, input_lengths, target_lengths
            # )
            return x, loss

        return x, None
