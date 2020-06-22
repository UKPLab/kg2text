"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import CGELWEncoder
from onmt.encoders.transformer import CGEEncoder
from onmt.encoders.transformer import PGELWEncoder
from onmt.encoders.transformer import PGEEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder


str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "cnn": CNNEncoder,
           "cge": CGEEncoder, "cge-lw": CGELWEncoder, "pge": PGEEncoder,
           "pge-lw": PGELWEncoder, "img": ImageEncoder,
           "audio": AudioEncoder, "mean": MeanEncoder}

__all__ = ["EncoderBase", "CGEEncoder", "CGELWEncoder", "PGEEncoder", "PGELWEncoder",
           "RNNEncoder", "CNNEncoder", "MeanEncoder", "str2enc"]
