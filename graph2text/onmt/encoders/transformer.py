"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask
from .utils_gnn import get_embs_graph
from .rgat import GATRConv
import torch


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0, gnn=None):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, mask, gnn_data=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, attns = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, attn_type="self")
        out = self.dropout(context) + inputs

        out_norm = self.ffn_layer_norm(out)
        outputs = self.feed_forward(out_norm)
        out = outputs + out

        return out, attns

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class CGELWEncoder(EncoderBase):
    def __init__(self, num_layers, d_model, d_dec, number_edge_types, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions):
        super(CGELWEncoder, self).__init__()

        self.d_model = d_model

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                self.d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

        self.gnn_dim = d_model
        self.dim_per_head_gnn = self.gnn_dim // heads
        self.heads = heads
        self.decoder_dim = d_dec

        self.gnns = nn.ModuleList()

        for i in range(num_layers):
            gnn = GATRConv(self.d_model, self.dim_per_head_gnn, num_relations=number_edge_types,
                     heads=heads, dropout=0.6)
            self.gnns.append(gnn)

        if self.decoder_dim != self.d_model:
            self.linear = nn.Linear(self.d_model, self.decoder_dim)
            self.linear_before = nn.Linear(self.decoder_dim, self.d_model)

        self.rnn = nn.GRUCell(self.gnn_dim, self.gnn_dim)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dec_rnn_size,
            opt.number_edge_types,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, lengths=None, batch=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        batch_size = emb.size()[1]
        seq_len = emb.size()[0]

        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)

        batch_geometric = get_embs_graph(batch, out)

        if self.decoder_dim != self.d_model:
            out = self.linear_before(out)

        for i, layer in enumerate(self.transformer):
            out, attns = layer(out, mask)

            out_gnn = out.view((batch_size * seq_len, self.d_model))
            memory_bank = self.gnns[i](out_gnn, batch_geometric.edge_index, edge_type=batch_geometric.y)

            out = self.rnn(memory_bank, out_gnn)

            out = out.view((batch_size, seq_len, self.d_model))

        out = self.layer_norm(out)
        if self.decoder_dim != self.d_model:
            out = self.linear(out)

        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)


class PGELWEncoder(EncoderBase):
    def __init__(self, num_layers, d_model, d_dec, number_edge_types, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions):
        super(PGELWEncoder, self).__init__()

        self.d_model = d_model

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                self.d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

        self.gnn_dim = d_model
        self.dim_per_head_gnn = self.gnn_dim // heads
        self.heads = heads
        self.decoder_dim = d_dec

        self.gnns = nn.ModuleList()

        for i in range(num_layers):
            gnn = GATRConv(self.d_model, self.dim_per_head_gnn, num_relations=number_edge_types,
                     heads=heads, dropout=0.6)
            self.gnns.append(gnn)

        if self.decoder_dim != self.d_model:
            self.linear_before = nn.Linear(self.decoder_dim, self.d_model)
            self.linear_after = nn.Linear(self.d_model, self.decoder_dim)

        self.linear = nn.Linear(self.d_model + self.gnn_dim, self.d_model)
        self.rnn = nn.GRUCell(self.gnn_dim, self.gnn_dim)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dec_rnn_size,
            opt.number_edge_types,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, lengths=None, batch=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        batch_size = emb.size()[1]
        seq_len = emb.size()[0]

        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)

        batch_geometric = get_embs_graph(batch, out)

        if self.decoder_dim != self.d_model:
            out = self.linear_before(out)

        for i, layer in enumerate(self.transformer):
            new_out, attn = layer(out, mask)

            out_gnn = out.view((batch_size * seq_len, self.d_model))
            memory_bank = self.gnns[i](out_gnn, batch_geometric.edge_index, edge_type=batch_geometric.y)
            memory_bank = memory_bank.view((batch_size, seq_len, self.d_model))

            new_out = torch.cat([new_out, memory_bank], dim=2)
            new_out = self.linear(new_out)

            new_out = new_out.view((batch_size * seq_len, self.d_model))
            out = out.view((batch_size * seq_len, self.d_model))
            out = self.rnn(new_out, out)
            out = out.view((batch_size, seq_len, self.d_model))

        out = self.layer_norm(out)
        if self.decoder_dim != self.d_model:
            out = self.linear_after(out)

        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)


class CGEEncoder(EncoderBase):
    def __init__(self, num_layers, local_layers, d_model, d_dec, number_edge_types, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions):
        super(CGEEncoder, self).__init__()

        self.d_model = d_model

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                self.d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

        self.gnn_dim = d_model
        self.dim_per_head_gnn = self.gnn_dim // heads
        self.heads = heads
        self.decoder_dim = d_dec

        self.gnns = nn.ModuleList()

        for i in range(local_layers):
            gnn = GATRConv(self.d_model, self.dim_per_head_gnn, num_relations=number_edge_types,
                     heads=heads, dropout=0.6)
            self.gnns.append(gnn)

        if self.decoder_dim != self.d_model:
            self.linear = nn.Linear(self.d_model, self.decoder_dim)
            self.linear_before = nn.Linear(self.decoder_dim, self.d_model)

        self.rnn = nn.GRUCell(self.gnn_dim, self.gnn_dim)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.local_layers,
            opt.enc_rnn_size,
            opt.dec_rnn_size,
            opt.number_edge_types,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, lengths=None, batch=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        batch_size = emb.size()[1]
        seq_len = emb.size()[0]

        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)

        if self.decoder_dim != self.d_model:
            out = self.linear_before(out)

        for layer in self.transformer:
            out, attns = layer(out, mask)
        out = self.layer_norm(out)

        batch_geometric = get_embs_graph(batch, out)
        memory_bank = batch_geometric.x

        for layer in self.gnns:
            new_memory_bank = layer(memory_bank, batch_geometric.edge_index, edge_type=batch_geometric.y)
            memory_bank = self.rnn(new_memory_bank, memory_bank)

        out = memory_bank.view((batch_size, seq_len, self.d_model))

        out = self.layer_norm(out)
        if self.decoder_dim != self.d_model:
            out = self.linear(out)

        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)


class PGEEncoder(EncoderBase):
    def __init__(self, num_layers, local_layers, d_model, d_dec, number_edge_types, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions):
        super(PGEEncoder, self).__init__()

        self.d_model = d_model

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                self.d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

        self.gnn_dim = d_model
        self.dim_per_head_gnn = self.gnn_dim // heads
        self.heads = heads
        self.decoder_dim = d_dec

        self.gnns = nn.ModuleList()

        for i in range(local_layers):
            gnn = GATRConv(self.d_model, self.dim_per_head_gnn, num_relations=number_edge_types,
                     heads=heads, dropout=0.6)
            self.gnns.append(gnn)

        if self.decoder_dim != self.d_model:
            self.linear_before = nn.Linear(self.decoder_dim, self.d_model)

        self.rnn = nn.GRUCell(self.gnn_dim, self.gnn_dim)
        self.final_linear = nn.Linear(self.d_model + self.gnn_dim, self.decoder_dim)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.local_layers,
            opt.enc_rnn_size,
            opt.dec_rnn_size,
            opt.number_edge_types,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, lengths=None, batch=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        batch_size = emb.size()[1]
        seq_len = emb.size()[0]

        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)

        if self.decoder_dim != self.d_model:
            out = self.linear_before(out)

        batch_geometric = get_embs_graph(batch, out)
        memory_bank = batch_geometric.x

        for layer in self.gnns:
            new_memory_bank = layer(memory_bank, batch_geometric.edge_index, edge_type=batch_geometric.y)
            memory_bank = self.rnn(new_memory_bank, memory_bank)

        memory_bank = memory_bank.view((batch_size, seq_len, self.d_model))

        for layer in self.transformer:
            out, attns = layer(out, mask)
        out = self.layer_norm(out)

        out = torch.cat([out, memory_bank], dim=2)

        out = self.final_linear(out)

        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)