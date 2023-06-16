import torch
from torch.nn.functional import ctc_loss, log_softmax
from torch.nn import Sequential, GRU, Module, AvgPool2d, Conv2d, Linear
import torch.nn as nn
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
    #
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
        hidden_size,
        num_classes,
    ):
        super(self.__class__, self).__init__()

        self.num_classes = num_classes

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.fc = Linear(in_features=hidden_size, out_features=num_classes)

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
        x = self.transformer_encoder(x)
        x = self.fc(x)

        return x


class CRNN(Module):
    def __init__(
        self,
        num_chars,
        cnn_input_size,
        cnn_output_len,
        rnn_hidden_size,
    ):
        super(self.__class__, self).__init__()

        self.num_chars = num_chars + 1
        self.features_extractor = FeatureExtractor(
            input_size=cnn_input_size, output_len=cnn_output_len
        )
        self.sequence_predictor = SequencePredictor(
            hidden_size=rnn_hidden_size,
            num_classes=self.num_chars,
        )

    def forward(self, images, targets=None, seq_len=None):
        batch_size, _, _, _ = images.size()

        x = self.features_extractor(images)
        # print(x.size())
        x = self.sequence_predictor(x)
        # x = x.permute(1, 0, 2)
        # print(x)
        # print(x.size())

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
