# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from VariationalDefinitionGeneration.model.layers import Embedding, WordEncoder
from VariationalDefinitionGeneration.model.BaseDecoder import BaseDecoder
from VariationalDefinitionGeneration.model.BaseModel import (
    BaseDGModel,
    set_default_args,
)
from VariationalDefinitionGeneration.model.baseline.RNNDG import RNNDG
import torch
import torch.nn.functional as F
import pdb


@register_model('twopass')
class TwoPassModel(FairseqEncoderDecoderModel):

    def __init__(self,first_pass_model,second_pass_model):
        super().__init__(first_pass_model.encoder,second_pass_model.decoder)
        self.first_pass_model = first_pass_model
        self.second_pass_model = second_pass_model

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--word-embed-path',type=str, metavar='STR')
        parser.add_argument('--encoder-freeze-embed', action='store_true',
                            help='freeze encoder embeddings')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-freeze-embed', action='store_true',
                            help='freeze decoder embeddings')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=bool, metavar='BOOL',
                            help='decoder attention')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        parser.add_argument("--rnn-type",type=str,metavar="STR")

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        set_default_args(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError("--encoder-layers must match --decoder-layers")

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        pretrained_word_embed = load_pretrained_embedding_from_file(
            args.word_embed_path, task.source_dictionary, args.encoder_embed_dim
        )
        pretrained_word_embed.requires_grad = False

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim
            )
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError("--share-all-embeddings requires a joint dictionary")
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embed not compatible with --decoder-embed-path"
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to "
                    "match --decoder-embed-dim"
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim,
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
            args.decoder_embed_dim != args.decoder_out_embed_dim
        ):
            raise ValueError(
                "--share-decoder-input-output-embeddings requires "
                "--decoder-embed-dim to match --decoder-out-embed-dim"
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        first_pass_encoder = RNNDG.build_encoder(task, args, task.source_dictionary, pretrained_encoder_embed, pretrained_word_embed,pretrained_decoder_embed)
        first_pass_decoder = RNNDG.build_decoder(task, args, task.target_dictionary, first_pass_encoder, pretrained_decoder_embed, pretrained_word_embed)

        second_pass_encoder = RNNDG.build_encoder(task, args, task.target_dictionary, pretrained_decoder_embed, pretrained_word_embed, pretrained_decoder_embed)
        second_pass_decoder = RNNDG.build_decoder(task, args, task.target_dictionary, second_pass_encoder,pretrained_decoder_embed,pretrained_word_embed)

        second_pass_encoder.embed_tokens.weight.data = first_pass_decoder.embed_tokens.weight.data
        second_pass_decoder.embed_tokens.weight.data = first_pass_decoder.embed_tokens.weight.data

        first_pass_model = RNNDG(args,first_pass_encoder,first_pass_decoder)
        second_pass_model = RNNDG(args,second_pass_encoder,second_pass_decoder)

        return cls(first_pass_model, second_pass_model)

    def forward(self,
                src_tokens,
                src_lengths,
                word_tokens,
                target1=None,
                target1_lengths=None,
                prev_output_tokens1=None,
                prev_output_tokens2=None):
        first_pass_output = self.first_pass_model(
            src_tokens,
            src_lengths,
            word_tokens,
            target=None,
            target_lengths=None,
            prev_output_tokens=prev_output_tokens1
        )
        second_pass_output = self.second_pass_model(
            word_tokens=word_tokens,
            src_tokens=target1,
            src_lengths=target1_lengths,
            target=None,
            target_lengths=None,
            prev_output_tokens=prev_output_tokens2
        )
        return (first_pass_output,second_pass_output)

    def get_first_pass_targets(self,sample,net_output):
        return sample['target1']

    def get_second_pass_targets(self,sample,net_output):
        return sample['target']

@register_model_architecture('twopass','twopass')
def set_default_args(args):
    args.dropout = getattr(args, "dropout", 0.5)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 300)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_freeze_embed = getattr(args, "encoder_freeze_embed", False)
    args.encoder_hidden_size = getattr(
        args, "encoder_hidden_size", args.encoder_embed_dim
    )
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_bidirectional = getattr(args, "encoder_bidirectional", True)
    args.encoder_dropout_in = getattr(args, "encoder_dropout_in", args.dropout)
    args.encoder_dropout_out = getattr(args, "encoder_dropout_out", args.dropout)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 300)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.decoder_hidden_size = getattr(
        args, "decoder_hidden_size", args.decoder_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 300)
    args.decoder_attention = getattr(args, "decoder_attention", "1")
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.rnn_type = getattr(args, "rnn_type", "lstm")
