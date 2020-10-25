import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

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
    WordEncoder,
    Linear,
    Embedding,
    AttentionLayer,
)

from VariationalDefinitionGeneration.model.BaseDecoder import BaseDecoder
from VariationalDefinitionGeneration.model.BaseModel import BaseDGModel,set_default_args

import pdb


class CVDGEncoder(FairseqEncoder):
    def __init__(
        self,
        src_dictionary,
        tgt_dictionary,
        latent_size,
        pretrained_word_embed,
        pretrained_x_encoder_embed,
        pretrained_y_encoder_embed,
        embed_dim,
        simple_posterior,
        **kwargs
    ):
        super(CVDGEncoder, self).__init__(src_dictionary)
        self.latent_size = latent_size
        self.simple_posterior = simple_posterior
        self.x_encoder = WordEncoder(
            pretrained_word_embed=pretrained_word_embed,
            pretrained_embed=pretrained_x_encoder_embed,
            dictionary=src_dictionary,
            embed_dim=embed_dim,
            **kwargs
        )

        self.prior_proj = nn.Linear(embed_dim, latent_size)
        self.prior_hidden2mean = nn.Linear(latent_size, latent_size)
        self.prior_hidden2logv = nn.Linear(latent_size, latent_size)

        if not self.simple_posterior:
            self.posterior_proj = nn.Linear(
                2*embed_dim, latent_size
            )
            self.y_encoder = WordEncoder(
                pretrained_word_embed=pretrained_word_embed,
                pretrained_embed=pretrained_y_encoder_embed,
                dictionary=tgt_dictionary,
                embed_dim=embed_dim,
                **kwargs
            )
            self.posterior_hidden2mean = nn.Linear(latent_size, latent_size)
            self.posterior_hidden2logv = nn.Linear(latent_size, latent_size)

    def forward(
        self, src_tokens, src_lengths, word_tokens, target=None, target_lengths=None
    ):
        enc_x = self.x_encoder(src_tokens, src_lengths, word_tokens)
        prior_mean, prior_logv, prior_z = self.compute_prior(enc_x["encoder_summary"])
        if target is not None and target_lengths is not None:
            if not self.simple_posterior:
                enc_y = self.y_encoder(target, target_lengths, word_tokens)
            posterior_mean, posterior_logv, posterior_z = self.compute_posterior(
                enc_x["encoder_summary"], enc_y["encoder_summary"] if not self.simple_posterior else None
            )
        else:
            enc_y, posterior_mean, posterior_logv, posterior_z = None, None, None, None
        return {
            "prior_mean": prior_mean,
            "prior_logv": prior_logv,
            "sampled_z": prior_z if self.simple_posterior else posterior_z,
            "encoder_padding_mask": enc_x["encoder_padding_mask"],
            "encoder_summary": enc_x["encoder_summary"],
            "words": enc_x["words"],
            "encoder_out": enc_x["encoder_out"],
            "posterior_mean": posterior_mean,
            "posterior_logv": posterior_logv,
        }

    def compute_prior(self, hidden):
        hidden = torch.tanh(self.prior_proj(hidden))
        mean = self.prior_hidden2mean(hidden)
        logv = self.prior_hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        if self.training:
            z = torch.randn_like(mean).to(hidden.device)
            z = z * std + mean
        else:
            z = mean
        return mean, logv, z

    def compute_posterior(self, hidden_x, hidden_y):
        if self.simple_posterior:
            mean = torch.zeros(hidden_x.size()[0],self.latent_size).to(hidden_x.device)
            logv = torch.zeros(hidden_x.size()[0],self.latent_size).to(hidden_x.device)
        else:
            hidden = torch.tanh(self.posterior_proj(torch.cat([hidden_x, hidden_y], -1)))
            mean = self.posterior_hidden2mean(hidden)
            logv = self.posterior_hidden2logv(hidden)

        std = torch.exp(0.5 * logv)
        z = torch.randn_like(mean).to(hidden_x.device)
        z = z * std + mean
        return mean, logv, z

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = tuple(
            eo.index_select(1, new_order) for eo in encoder_out["encoder_out"]
        )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(1, new_order)
        encoder_out["encoder_summary"] = encoder_out["encoder_summary"].index_select(
            0, new_order
        )
        encoder_out["words"] = encoder_out["words"].index_select(0, new_order)
        encoder_out["prior_mean"] = encoder_out["prior_mean"].index_select(0, new_order)
        encoder_out["prior_logv"] = encoder_out["prior_logv"].index_select(0, new_order)
        encoder_out["sampled_z"] = encoder_out["sampled_z"].index_select(0, new_order)
        return encoder_out


class GussianDecoder(BaseDecoder):
    """Gussian decoder."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_latent(self, encoder_out):
        if self.training:
            if "sampled_z" not in encoder_out:
                raise KeyError
            latent = encoder_out["sampled_z"]
        else:
            if "prior_mean" not in encoder_out:
                raise KeyError
            latent = encoder_out["prior_mean"]
        return latent

    def gather_outout(self, attn_scores, encoder_out):
        return {
            "posterior_mean": encoder_out["posterior_mean"]
            if "posterior_mean" in encoder_out
            else None,
            "posterior_logv": encoder_out["posterior_logv"]
            if "posterior_logv" in encoder_out
            else None,
            "sampled_z": encoder_out["sampled_z"]
            if "sampled_z" in encoder_out
            else None,
            "prior_mean": encoder_out["prior_mean"],
            "prior_logv": encoder_out["prior_logv"],
            "attn_scores": attn_scores,
        }


@register_model("CVDG")
class GussianVIModel(BaseDGModel):
    def __init__(self, args,encoder, decoder):
        super().__init__(args,encoder, decoder)
        if getattr(args,'ppmi_loss',False):
            self.ppmi_proj = nn.Sequential(
                Linear(self.encoder.latent_size, self.decoder.embed_dim), nn.Tanh()
            )
        elif getattr(args,'bow_loss',False):
            self.bow_proj = nn.Sequential(
                Linear(self.encoder.latent_size,self.decoder.embed_dim),
                nn.ReLU(),
            )

        self.args = args

    @staticmethod
    def add_args(parser):
        BaseDGModel.add_args(parser)
        parser.add_argument("--latent-size", type=int, default=300)
        parser.add_argument("--ppmi-loss",action='store_true',)
        parser.add_argument("--bow-loss",action='store_true',)
        parser.add_argument('--simple-posterior',action='store_true')

    @classmethod
    def build_encoder(cls, task, args, pretrained_encoder_embed, pretrained_word_embed,pretrained_decoder_embed):
        encoder = CVDGEncoder(
            src_dictionary=task.source_dictionary,
            tgt_dictionary=task.target_dictionary,
            latent_size=args.latent_size,
            pretrained_word_embed=pretrained_word_embed,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_x_encoder_embed=pretrained_encoder_embed,
            pretrained_y_encoder_embed=pretrained_decoder_embed,
            simple_posterior=args.simple_posterior
        )
        return encoder

    @classmethod
    def build_decoder(
            cls, task, args, encoder, pretrained_decoder_embed, pretrained_word_embed
    ):
        decoder = GussianDecoder(
            latent_size=args.latent_size,
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.x_encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == "adaptive_loss"
                else None
            ),
            rnn_type=args.rnn_type,
        )
        return decoder

    def forward(
        self,
        word_tokens,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        target,
        target_lengths,
    ):
        encoder_out = self.encoder(
            src_tokens, src_lengths, word_tokens, target, target_lengths
        )
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out)
        return decoder_out

    def get_kld_loss(self, net_output):
        mu_1 = net_output[1]["posterior_mean"]
        logv_1 = net_output[1]["posterior_logv"]
        mu_2 = net_output[1]["prior_mean"]
        logv_2 = net_output[1]["prior_logv"]

        prior_distribution = MultivariateNormal(
            mu_2, covariance_matrix=torch.diag_embed(logv_2.exp())
        )
        posterior_distribution = MultivariateNormal(
            mu_1, covariance_matrix=torch.diag_embed(logv_1.exp())
        )

        if not self.encoder.simple_posterior:
            kld = kl_divergence(posterior_distribution, prior_distribution)
        else:
            kld = kl_divergence(prior_distribution,posterior_distribution)
        return kld

    def get_ppmi_loss(self, net_output, sample):
        # if not self.training:
        #     pdb.set_trace()
        # pdb.set_trace()
        z = net_output[1]["sampled_z"]  # bsize * latent_size
        target_tokens = sample["net_input"]["target"]  # bsize * seqlen
        target_lengths = sample["net_input"]["target_lengths"].float()
        target_mask = target_tokens.eq(self.decoder.padding_idx)
        ppmi = sample["ppmi"]  # bsize * seqlen
        target_emb = self.decoder.embed_tokens(
            target_tokens
        )  # bsize * seqlen * D

        z = self.ppmi_proj(z).unsqueeze(1).expand_as(target_emb)  # bsize * seqlen * D
        predicted_ppmi = torch.mul(z, target_emb).sum(-1)  # bsize * seqlen
        ppmi_loss = torch.pow(ppmi-predicted_ppmi,2).masked_fill(target_mask,0.0).sum(-1)
        return ppmi_loss

    def get_bow_loss(self,net_output,sample):
        z = net_output[1]["sampled_z"]  # bsize * D
        bow = sample['bow']
        nbow = sample['nbow']
        bow_mask = bow.eq(self.decoder.padding_idx)
        nbow_mask = nbow.eq(self.decoder.padding_idx)
        bow_emb = self.bow_proj(self.decoder.embed_tokens(bow).detach())  # bsize * bow_len * D
        nbow_emb = self.bow_proj(self.decoder.embed_tokens(nbow).detach())  # bsize * n_neg * D
        oloss = torch.nn.functional.logsigmoid(torch.bmm(bow_emb,z.unsqueeze(2)).squeeze()) # bsize * bow_len
        oloss = oloss.masked_fill(bow_mask,0.0).sum(-1)
        nloss = torch.nn.functional.logsigmoid(-torch.bmm(nbow_emb,z.unsqueeze(2)).squeeze()).masked_fill(nbow_mask,0.0).sum(-1)
        return -(oloss + nloss).sum()

    def decode_z(self, sample, k=10):
        # pdb.set_trace()
        net_input = sample["net_input"]
        with torch.no_grad():
            encoder_out = self.encoder(
                net_input["src_tokens"],
                net_input["src_lengths"],
                net_input["word_tokens"],
                target=net_input["target"],
                target_lengths=net_input["target_lengths"],
            )
            target_emb = self.decoder.embed_tokens.weight.unsqueeze(0)
            gold_ppmi = sample["ppmi"]

            prior_z = encoder_out["prior_z"]
            prior_z = self.ppmi_proj(prior_z).unsqueeze(1)  # bsize * D
            predicted_prior_ppmi = torch.mul(prior_z, target_emb).sum(-1)
            prior_topk_score, prior_topk_idx = predicted_prior_ppmi.topk(
                k, dim=-1, largest=True, sorted=True
            )

            posterior_z = encoder_out["posterior_z"]
            posterior_z = self.ppmi_proj(posterior_z).unsqueeze(1)
            predicted_posterior_ppmi = torch.mul(posterior_z, target_emb).sum(-1)
            posterior_topk_score, posterior_topk_idx = predicted_posterior_ppmi.topk(
                k, dim=-1, largest=True, sorted=True
            )

        return prior_topk_score, prior_topk_idx


@register_model_architecture("CVDG", "CVDG")
def base_architecture(args):
    set_default_args(args)
    args.latent_size = getattr(args, "latent_size", 300)

@register_model_architecture('CVDG','Lite-CVDG')
def lite_cvdg(args):
    base_architecture(args)
    args.simple_posterior = getattr(args,'simple_posterior',True)
