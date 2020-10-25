import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax
import pdb


class WordEncoder(FairseqEncoder):
    """LSTM encoder."""

    def __init__(
        self,
        pretrained_word_embed,
        char_embed,
        dictionary,
        embed_dim,
        hidden_size,
        num_layers,
        dropout_in,
        dropout_out,
        bidirectional,
        pretrained_embed,
        rnn_type="lstm",
        padding_value=0.0,
        left_pad=True,
        using_bert=False
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.using_bert = using_bert

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()

        self.char_embed = char_embed
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        RNN = LSTM if rnn_type == "lstm" else GRU
        self.rnn = RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

        self.embed_words = pretrained_word_embed
        self.summary_network = SummaryNetwork(self.output_units,embed_dim)

    def forward(self,src_tokens,src_lengths,word_tokens=None,bert_repre=None,**kwargs):

        words = None
        chars = None
        if word_tokens is not None and self.embed_words is not None:
            words = self.embed_words(word_tokens).squeeze().detach()
            if self.char_embed is not None:
                chars = self.char_embed(word_tokens).squeeze()

        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens, self.padding_idx, left_to_right=True
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths.data.tolist(), enforce_sorted=False
        )

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        if self.rnn_type == "lstm":
            c0 = x.new_zeros(*state_size)
            packed_outs, _ = self.rnn(packed_x, (h0, c0))
        else:
            packed_outs, _ = self.rnn(packed_x, h0)

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_value
        )
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        if word_tokens is not None:
            encoder_summary = self.summary_network(x, words, encoder_padding_mask)
        else:
            encoder_summary = None
        return {
            "encoder_out": (x,),
            "encoder_padding_mask": encoder_padding_mask
            if encoder_padding_mask.any()
            else None,
            "encoder_summary": encoder_summary,
            "words": words,
            "chars": chars,
            'bert_repre': bert_repre
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = tuple(
            eo.index_select(1, new_order) for eo in encoder_out["encoder_out"]
        )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(1, new_order)
        if encoder_out["encoder_summary"] is not None:
            encoder_out["encoder_summary"] = encoder_out[
                "encoder_summary"
            ].index_select(0, new_order)
        if encoder_out["words"] is not None:
            encoder_out["words"] = encoder_out["words"].index_select(0, new_order)
        if encoder_out["chars"] is not None:
            encoder_out["chars"] = encoder_out["chars"].index_select(0, new_order)
        if encoder_out['bert_repre'] is not None:
            encoder_out['bert_repre'] = encoder_out['bert_repre'].index_select(0,new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class SememeEncoder(nn.Module):

    def __init__(self,sememe_dim,hidden_size,padding_value=0.0):
        super(SememeEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = LSTM(
            input_size=sememe_dim,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.0,
            bidirectionarl=True
        )

    def forward(self,sememe_emb,src_lengths):
        bsz,seqlen,D = sememe_emb.size()

        # B x T x D -> T x B x D
        x = sememe_emb.transpose(0,1)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            x,src_lengths.data.tolist(),enforce_sorted=False
        )

        state_size = 2, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs,_ = self.rnn(packed_x,(h0,c0))

        x,_ = nn.utils.rnn.pad_packed_sequence(
            packed_outs,padding_value=self.padding_value
        )
        x = x.transpose(0,1)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim,bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(source_embed_dim,output_embed_dim,bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = (
                attn_scores.float()
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(attn_scores)
            )  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)
        x = self.output_proj(x)
        return x, attn_scores


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def GRU(input_size, hidden_size, **kwargs):
    m = nn.GRU(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def GRUCell(input_size, hidden_size, **kwargs):
    m = nn.GRUCell(input_size, hidden_size)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


class LatentGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LatentGRUCell, self).__init__()
        self.ru_x = Linear(input_size, 2 * hidden_size)
        self.ru_h = Linear(hidden_size, 2 * hidden_size)
        self.ru_z = Linear(hidden_size, 2 * hidden_size)

        self.s_x = Linear(input_size, hidden_size)
        self.s_h = Linear(hidden_size, hidden_size)
        self.s_z = Linear(hidden_size, hidden_size)

    def forward(self, x, h, z=None):
        if z is not None:
            ru = self.ru_x(x) + self.ru_h(h) + self.ru_z(z)
        else:
            ru = self.ru_x(x) + self.ru_h(h)
        r, u = torch.split(ru, ru.size(1) // 2, dim=-1)
        r, u = torch.sigmoid(r), torch.sigmoid(u)

        if z is not None:
            s = self.s_x(x) + self.s_h(torch.mul(r, h)) + self.s_z(z)
        else:
            s = self.s_x(x) + self.s_h(torch.mul(r, h))
        s = torch.tanh(s)

        return torch.mul(1 - u, h) + torch.mul(u, s)


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if "weight" in name or "bias" in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


class GateCombineNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(GateCombineNetwork, self).__init__()
        self.proj1 = Linear(hidden_size, hidden_size)
        self.proj2 = Linear(hidden_size, hidden_size)

    def forward(self, a, b):
        gate = torch.sigmoid(self.proj1(a) + self.proj2(b))
        return torch.mul(gate, a) + torch.mul(1 - gate, b)


class GateUpdateNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GateUpdateNetwork, self).__init__()
        # self.input_proj = Linear(input_size, hidden_size)
        self.Wz = Linear(hidden_size + input_size, hidden_size)
        self.Wr = Linear(hidden_size + input_size, input_size)
        self.Ws = Linear(hidden_size + input_size, hidden_size)

    def forward(self, ft, st_1):
        # input = torch.tanh(self.input_proj(ft))
        zt = torch.sigmoid(self.Wz(torch.cat([ft, st_1], dim=-1)))
        rt = torch.sigmoid(self.Wr(torch.cat([ft, st_1], dim=-1)))
        st_hat = torch.tanh(self.Ws(torch.cat([torch.mul(rt, ft), st_1], dim=-1)))
        st = torch.mul(1 - zt, st_1) + torch.mul(zt, st_hat)
        return st


class SummaryNetwork(nn.Module):
    def __init__(self, hidden_size,embed_dim):
        super(SummaryNetwork, self).__init__()
        self.embed_dim = embed_dim
        self.ann = Linear(hidden_size, embed_dim)
        self.W = Linear(embed_dim,embed_dim)

    def forward(self, x, w, x_mask=None):
        x_mask = ~x_mask.unsqueeze(-1)
        x_lengths = x_mask.float().sum(0)
        seqlen, bsize, d = x.size()
        x = self.ann(x.view(bsize * seqlen, d)).view(seqlen, bsize, self.embed_dim)
        if x_mask is not None:
            x = (x * x_mask.float()).sum(0) / x_lengths
        else:
            x = x.mean(0)
        gate = torch.sigmoid(self.W(x))
        return torch.mul(gate, w)


class Hidden2Discrete(nn.Module):
    def __init__(self,input_size,latent_M,latent_K):
        super(Hidden2Discrete, self).__init__()
        self.latent_M = latent_M
        self.latent_K = latent_K
        latent_size = latent_K * latent_M
        self.p_h = nn.Sequential(
            nn.Linear(input_size,input_size),
            nn.ReLU(),
            nn.Linear(latent_size,latent_size)
        )

    def forward(self,inputs):
        logits = self.p_h(inputs).view(-1,self.latent_K)
        log_qy = F.log_softmax(logits,dim=1)
        return logits, log_qy


class SimpleSentEncoder(nn.Module):
    def __init__(self,embed,unigram_p,encoder_type='bow'):
        super(SimpleSentEncoder, self).__init__()
        self.unigram_p = unigram_p
        if encoder_type == 'unigram-bow' or encoder_type == 'SIF':
            self.gamma = 1e-4
        self.encoder_type = encoder_type
        self.target_embed = embed
        # vocab_size = decoder_embed.weight.size()[0]
        # embed_dim = decoder_embed.weight.size()[1]
        # self.embedding = nn.embedding(vocab_size,embed_dim)

    def forward(self,target,target_length):
        target_emb = self.target_embed(target)
        if self.encoder_type == 'bow':
            vs = target_emb.sum(dim=1) / target_length.unsqueeze(1).float()
        elif self.encoder_type == 'unigram-bow':
            weight = self.gamma / (self.gamma + F.embedding(target,self.unigram_p))
            vs = (weight.unsqueeze(-1) * target_emb).sum(dim=1) / target_length.unsqueeze(1).float()  # b * d
        elif self.encoder_type == 'SIF':
            weight = self.gamma / (self.gamma + F.embedding(target,self.unigram_p))
            vs = (weight.unsqueeze(-1) * target_emb).sum(dim=1) / target_length.unsqueeze(1).float()  # b * d
            U,S,V = torch.svd(vs.t())
            singular_vec = -U[:,0]   # D
            vs = vs - torch.mm(vs,singular_vec.unsqueeze(1)) * singular_vec
        return vs


def pairwise_cosine_similarity(a,b=None,eps=1e-8):
    if b is None:
        b = a
    a_n,b_n = a.norm(dim=-1)[:,:,None], b.norm(dim=-1)[:,:,None]
    a_norm = a / torch.max(a_n,eps*torch.ones_like(a_n))
    b_norm = b / torch.max(b_n,eps*torch.ones_like(b_n))
    sim_mt = torch.bmm(a_norm,b_norm.transpose(-2,-1))
    return sim_mt


def test_simple_sent_encoder():
    unigram_p = torch.randn(500)
    gamma = 1e-3
    decoder_embed = nn.Embedding(500,20)
    encoder = SimpleSentEncoder(decoder_embed,unigram_p,gamma)
    target = torch.randint(low=0,high=499,size=(8,20))
    target_length = torch.randint(low=0,high=20,size=(8,))
    print(encoder(target,target_length))
