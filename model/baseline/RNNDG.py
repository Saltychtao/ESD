# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from VariationalDefinitionGeneration.model.layers import Embedding, WordEncoder
from VariationalDefinitionGeneration.model.BaseDecoder import BaseDecoder
from VariationalDefinitionGeneration.model.BaseModel import (
    BaseDGModel,
    set_default_args,
)
import pdb


class RNNDecoder(BaseDecoder):
    def prepare_latent(self, encoder_out):
        return None

    def gather_outout(self, attn_scores, encoder_out):
        return {"attn_scores": attn_scores}


@register_model("RNNDG")
class RNNDG(BaseDGModel):
    def __init__(self, args,encoder, decoder):
        super().__init__(args,encoder, decoder)

    @classmethod
    def build_encoder(cls, task, args, dictionary):
        encoder = WordEncoder(
            pretrained_word_embed=args.pretrained_word_embed,
            char_embed=args.char_embed if hasattr(args,'char_embed') else None,
            dictionary=dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=args.pretrained_encoder_embed,
            rnn_type=args.rnn_type,
        )
        return encoder

    @classmethod
    def build_decoder(cls, task, args, dictionary,encoder):
        decoder = RNNDecoder(
            dictionary=dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.output_units,
            pretrained_embed=args.pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == "adaptive_loss"
                else None
            ),
            rnn_type=args.rnn_type,
            info_type=args.info_type
        )
        return decoder

    def forward(self,src_tokens,src_lengths,word_tokens,prev_output_tokens,target=None,target_lengths=None,**kwargs):
        encoder_out = self.encoder(src_tokens,src_lengths,word_tokens,**kwargs)
        decoder_out = self.decoder(prev_output_tokens,encoder_out,**kwargs)
        return decoder_out


@register_model_architecture("RNNDG", "RNNDG")
def base_architecture(args):
    set_default_args(args)
    args.info_type = 'context,word,char'
