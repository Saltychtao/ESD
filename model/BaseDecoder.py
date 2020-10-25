import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax
from VariationalDefinitionGeneration.model.layers import (
    GateCombineNetwork,
    GateUpdateNetwork,
    LSTM,
    LSTMCell,
    GRUCell,
    GRU,
    Embedding,
    Linear,
    AttentionLayer,
    WordEncoder,
)


class BaseDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""

    def __init__(
        self,
        dictionary,
        embed_dim,
        hidden_size,
        out_embed_dim,
        num_layers,
        dropout_in,
        dropout_out,
        attention,
        encoder_output_units,
        pretrained_embed,
        share_input_output_embed,
        adaptive_softmax_cutoff,
        rnn_type,
        info_type
    ):
        super().__init__(dictionary)
        self.embed_dim = embed_dim
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.rnn_type = rnn_type
        self.info_type = info_type

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.padding_idx = padding_idx
        self.vocab_size = num_embeddings
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size:
            self.encoder_hidden_proj = Linear(embed_dim, hidden_size)
            self.encoder_cell_proj = Linear(embed_dim, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None
        cell = LSTMCell if rnn_type == "lstm" else GRUCell
        self.layers = nn.ModuleList(
            [
                cell(
                    input_size=embed_dim if layer == 0 else hidden_size,
                    hidden_size=hidden_size,
                )
                for layer in range(num_layers)
            ]
        )
        if attention:
            self.attention = AttentionLayer(
                hidden_size, encoder_output_units, hidden_size,bias=False
            )
        else:
            self.attention = None
        info_dim = hidden_size
        if 'word' in self.info_type:
            info_dim += embed_dim
        if 'char' in self.info_type:
            info_dim += 160
        if 'bert' in self.info_type:
            info_dim += 768
        if 'latent' in self.info_type:
            info_dim += hidden_size
        self.gate_update_network = GateUpdateNetwork(
            info_dim,
            hidden_size,
        )
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(
                num_embeddings,
                hidden_size,
                adaptive_softmax_cutoff,
                dropout=dropout_out,
            )
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def prepare_latent(self, encoder_out):
        raise NotImplementedError

    def forward(
        self, prev_output_tokens, encoder_out, incremental_state=None
    ):
        encoder_padding_mask = encoder_out["encoder_padding_mask"]

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs = encoder_out["encoder_out"][0]
        encoder_summary = encoder_out['encoder_summary']
        words = encoder_out["words"]  # bsize * D
        chars = encoder_out['chars']
        bert_repre = encoder_out['bert_repre']
        srclen = encoder_outs.size(0)

        # pdb.set_trace()
        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)  # bsize * T * D

        ## preappend the word embedding when training
        if incremental_state is None:
            x = torch.cat([words.unsqueeze(1), x], dim=1)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(
            self, incremental_state, "cached_state"
        )
        if cached_state is not None:
            if self.rnn_type == "lstm":
                prev_hiddens, prev_cells, input_feed = cached_state
            else:
                prev_hiddens, input_feed = cached_state
        else:
            num_layers = len(self.layers)
            prev_hiddens = [torch.zeros_like(words) for i in range(num_layers)]
            if self.rnn_type == "lstm":
                prev_cells = [torch.zeros_like(words) for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                if self.rnn_type == "lstm":
                    prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)

        latent = self.prepare_latent(encoder_out)

        attn_scores = x.new_zeros(srclen, seqlen, bsz)
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = x[j, :, :]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                if self.rnn_type == "lstm":
                    hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
                else:
                    hidden = rnn(input, prev_hiddens[i])

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                context_vector, attn_scores[:, j, :] = self.attention(
                    hidden, encoder_outs, encoder_padding_mask
                )
                info = [context_vector]
                if 'word' in self.info_type:
                    info.append(words)
                if 'char' in self.info_type:
                    info.append(chars)
                if 'bert' in self.info_type:
                    info.append(bert_repre)
                if latent is not None:
                    info.append(latent)

                f_t = torch.cat(info, dim=-1)
                hidden = self.gate_update_network(f_t, hidden)
                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out = F.dropout(hidden, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self,
            incremental_state,
            "cached_state",
            (prev_hiddens, prev_cells, input_feed)
            if self.rnn_type == "lstm"
            else (prev_hiddens, input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        # project back to size of vocabulary
        if self.adaptive_softmax is None:
            if hasattr(self, "additional_fc"):
                x = self.additional_fc(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        net_output = self.gather_outout(attn_scores, encoder_out)
        return x, net_output

    def gather_outout(self, attn_scores):
        raise NotImplementedError

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(
            self, incremental_state, "cached_state"
        )
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, "cached_state", new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn
