import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import random
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
    WordEncoder,
    Linear,
    Embedding,
    AttentionLayer,
    Hidden2Discrete,
    pairwise_cosine_similarity,
    SimpleSentEncoder
)

from VariationalDefinitionGeneration.model.BaseDecoder import BaseDecoder
from VariationalDefinitionGeneration.model.BaseModel import BaseDGModel,set_default_args


def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
    num_embedding = len(dictionary)
    padding_idx = dictionary.pad()
    embed_tokens = Embedding(num_embedding, embed_dim, padding_idx)
    embed_dict = utils.parse_embedding(embed_path)
    utils.print_embed_overlap(embed_dict, dictionary)
    return utils.load_embedding(embed_dict, dictionary, embed_tokens)


class DVDGEncoder(FairseqEncoder):
    def __init__(
        self,
        src_dictionary,
        latent_K,
        latent_M,
        pretrained_word_embed,
        pretrained_x_encoder_embed,
        embed_dim,
        char_embed,
        using_gold_sememe=False,
        using_bert=False,
        pretrained_sememe_embed=None,
        **kwargs
    ):
        super(DVDGEncoder, self).__init__(src_dictionary)
        self.latent_M = latent_M
        self.latent_K = latent_K
        self.embed_dim = embed_dim
        self.using_gold_sememe = using_gold_sememe
        self.pad_idx = src_dictionary.pad()
        self.x_encoder = WordEncoder(
            pretrained_word_embed=pretrained_word_embed,
            pretrained_embed=pretrained_x_encoder_embed,
            dictionary=src_dictionary,
            embed_dim=embed_dim,
            char_embed=char_embed,
            **kwargs
        )

        self.using_bert = using_bert

        # self.z_embedding = Linear(latent_M*latent_K,embed_dim,bias=False)
        if self.using_gold_sememe:
            self.z_embedding = Linear(latent_K,embed_dim,bias=False)
            if pretrained_sememe_embed is not None:
                self.z_embedding.weight.data = pretrained_sememe_embed.weight.data.transpose(0,1)
            with torch.no_grad():
                self.z_embedding.weight.data[self.pad_idx].fill_(0.)
        else:
            self.p_h = nn.Sequential(
                Linear(embed_dim if not using_bert else 768+embed_dim,int(0.5*latent_M*latent_K)),
                nn.Tanh(),
                Linear(int(0.5*latent_K*latent_M),latent_M*latent_K),
                nn.Softplus()
            )
            self.z_embedding = Linear(latent_M*latent_K,embed_dim,bias=False)
            with torch.no_grad():
                for i in range(latent_M):
                    self.z_embedding.weight.data[:,self.pad_idx + i*latent_K].fill_(0.)

    def forward(
            self, src_tokens, src_lengths, word_tokens, target=None, target_lengths=None,bert_repre=None,bag_of_sememe=None
    ):
        bsize = src_tokens.size()[0]
        enc_x = self.x_encoder(src_tokens, src_lengths, word_tokens)

        # log_qy = F.log_softmax(logits_qy,dim=-1)
        # log_py = torch.log(torch.ones_like(log_qy)/self.latent_K)

        # when gold sememe is given, we use the gold sememe
        if self.using_gold_sememe:
            bsize,latent_M = bag_of_sememe.size()
            sampled_z = torch.zeros(bsize,latent_M,self.latent_K).to(src_tokens.device).scatter_(-1, bag_of_sememe.unsqueeze(-1), 1.0).view(bsize,latent_M,self.latent_K)
            latent_mask = bag_of_sememe != self.pad_idx
            # # map the code to semantic space
            latent_context = torch.bmm(sampled_z,self.z_embedding.weight.transpose(0,1).unsqueeze(0).expand(bsize,self.latent_K,self.embed_dim))  # bsize x latent_M x D
        # when training a latent variable model, we use Gumbel Softmax
        else:
            logits_qy = torch.log(self.p_h(enc_x['words']).view(-1,self.latent_K) + 1e-8)
            if self.training:
                sampled_z = F.gumbel_softmax(logits_qy,hard=True).view(bsize,self.latent_M,self.latent_K)
        # when testing, we use argmax to get the code
            else:
                index = logits_qy.max(dim=-1, keepdim=True)[1]
                sampled_z = torch.zeros_like(logits_qy).scatter_(-1, index, 1.0).view(bsize,self.latent_M,self.latent_K)

            latent_mask = torch.ones(bsize,self.latent_M).to(sampled_z.device).bool()
            z_embeddings = torch.t(self.z_embedding.weight).split(self.latent_K, dim=0)
            latent_context = []
            temp_sampled_z = sampled_z.view(-1, self.latent_M, self.latent_K)
            for z_id in range(self.latent_M):
                latent_context.append(torch.mm(temp_sampled_z[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            latent_context = torch.cat(latent_context, dim=1)

        return {
            "encoder_padding_mask": enc_x["encoder_padding_mask"],
            "encoder_summary": enc_x["encoder_summary"],
            "words": enc_x["words"],
            'chars': enc_x['chars'],
            "encoder_out": enc_x["encoder_out"],
            # 'log_py': log_py.view(bsize,self.latent_M,self.latent_K),
            # 'log_qy': log_qy.view(bsize,self.latent_M,self.latent_K),
            'sampled_z': sampled_z,
            'latent_mask': latent_mask,
            'latent_context': latent_context,
            'bert_repre': bert_repre
        }

    def compute_z_code(self,enc_x,bert_repre=None):
        if self.using_bert:
            logits_qy = torch.log(
                self.p_h(
                    torch.cat([bert_repre,enc_x['words']],dim=-1)
                ).view(-1,self.latent_K) + 1e-8
            )
        else:
            logits_qy = torch.log(self.p_h(enc_x['words']).view(-1,self.latent_K) + 1e-8)
        # logits_qy = self.p_h(enc_x['encoder_summary']).view(-1,self.latent_K)
        # logits_qy = torch.log(self.p_h(enc_x['words']).view(-1,self.latent_K) + 1e-8)
        # logits_qy, log_qy = self.x2z(enc_x['encoder_summary'])  # logits : (bsize * latent_M) x latent_K
        if self.training:
            sampled_z = F.gumbel_softmax(logits_qy).view(-1,self.latent_M,self.latent_K)  #
        else:
            index = logits_qy.max(dim=-1, keepdim=True)[1]
            sampled_z = torch.zeros_like(logits_qy).scatter_(-1, index, 1.0).view(-1,self.latent_M,self.latent_K).argmax(-1)
        return sampled_z

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
        encoder_out['chars'] = encoder_out['chars'].index_select(0,new_order)
        if encoder_out['bert_repre'] is not None:
            encoder_out['bert_repre'] = encoder_out['bert_repre'].index_select(0,new_order)
        # encoder_out['log_py'] = encoder_out['log_py'].index_select(0,new_order)
        # encoder_out['log_qy'] = encoder_out['log_qy'].index_select(0,new_order)
        encoder_out['sampled_z'] = encoder_out['sampled_z'].index_select(0,new_order)
        encoder_out['latent_context'] = encoder_out['latent_context'].index_select(0,new_order)
        encoder_out['latent_mask'] = encoder_out['latent_mask'].index_select(0,new_order)
        return encoder_out


class CategoricalDecoder(BaseDecoder):
    """Gussian decoder."""

    def __init__(self,latent_K,latent_M,**kwargs):
        super(CategoricalDecoder, self).__init__(**kwargs)
        self.latent_K = latent_K
        self.latent_M = latent_M
        self.latent_attention = AttentionLayer(
            kwargs['hidden_size'], kwargs['embed_dim'], kwargs['embed_dim'],bias=False
        )

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
        latent_context = encoder_out['latent_context'].transpose(0,1)
        srclen = encoder_outs.size(0)
        latent_M,bsize,D = latent_context.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)  # bsize * T * D

        # ## preappend the word embedding
        # x = torch.cat([words.unsqueeze(1), x], dim=1)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        latent = self.prepare_latent(encoder_out)
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
            prev_hiddens = [words for i in range(num_layers)]
            if self.rnn_type == "lstm":
                prev_cells = [torch.zeros_like(words) for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                if self.rnn_type == "lstm":
                    prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)

        attn_scores = x.new_zeros(srclen, seqlen, bsz)
        latent_attn_scores = x.new_zeros(latent_M,seqlen,bsz)
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
                latent_context_vector, latent_attn_scores[:,j,:] = self.latent_attention(
                    hidden, latent_context, ~(encoder_out['latent_mask'].transpose(0,1)) if 'latent_mask' in encoder_out else None)
                info = [context_vector]
                if 'word' in self.info_type:
                    info.append(words)
                if 'char' in self.info_type:
                    info.append(chars)
                if 'bert' in self.info_type:
                    info.append(bert_repre)
                if latent is not None:
                    info.append(latent_context_vector)
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

    def prepare_latent(self, encoder_out):
        if "latent_context" not in encoder_out:
            raise KeyError
        latent = encoder_out["latent_context"].sum(1)  # bsize * latent_K
        return latent

    def gather_outout(self, attn_scores, encoder_out):
        return {
            # "log_qy": encoder_out["log_qy"]
            # if "log_qy" in encoder_out
            # else None,
            # "log_py": encoder_out["log_py"]
            # if "log_py" in encoder_out
            # else None,
            "sampled_z": encoder_out["sampled_z"]
            if "sampled_z" in encoder_out
            else None,
            "attn_scores": attn_scores,
            'latent_context': encoder_out['latent_context'],
            'words': encoder_out['words'],
            'encoder_summary': encoder_out['encoder_summary'],
        }


@register_model("DVDG")
class DiscreteVIModel(BaseDGModel):
    def __init__(self, args,encoder, decoder):
        super().__init__(args,encoder, decoder)

        self.args = args
        self.using_gold_sememe = args.using_gold_sememe

        if args.y_encoder_type == 'BILSTM' and args.target_type == 'definition':
            self.y_encoder = WordEncoder(pretrained_word_embed=args.pretrained_word_embed,
                                         pretrained_embed=args.pretrained_decoder_embed,
                                         dictionary=args.target_dictionary,
                                         embed_dim=args.encoder_embed_dim,
                                         hidden_size=args.encoder_hidden_size,
                                         num_layers=args.encoder_layers,
                                         dropout_in=args.encoder_dropout_in,
                                         dropout_out=args.encoder_dropout_out,
                                         bidirectional=args.encoder_bidirectional,
                                         char_embed=args.char_embed if hasattr(args,'char_embed') else None)
            self.y_encoder.embed_tokens.weight.data = self.decoder.embed_tokens.weight.data
            self.y_encoder.embed_words.weight.data = self.encoder.x_encoder.embed_words.weight.data
        elif args.y_encoder_type in ['SIF','bow','unigram-bow'] and args.target_type == 'definition':
            dictionary = args.target_dictionary
            unigram_p = torch.Tensor(dictionary.count).cuda()
            unigram_p = unigram_p / unigram_p.sum()
            for i in range(dictionary.nspecial):
                unigram_p[i] = 1
            self.y_encoder = SimpleSentEncoder(self.decoder.embed_tokens,unigram_p,args.y_encoder_type)

        elif args.y_encoder_type == 'BOW':
            # if args.target_type == 'definition':
            #     dictionary = args.target_dictionary
            #     unigram_p = torch.Tensor(dictionary.count).cuda()
            #     unigram_p = unigram_p / unigram_p.sum()
            #     for i in range(dictionary.nspecial):
            #         unigram_p[i] = 1
            #     self.unigram_p = unigram_p
            if args.target_type == 'sememe':
                self.sememe_embed = args.sememe_embed

    @staticmethod
    def add_args(parser):
        BaseDGModel.add_args(parser)
        parser.add_argument("--latent-M", type=int, default=20)
        parser.add_argument('--latent-K',type=int,default=100)
        parser.add_argument('--decoder-init',type=str,default='word')
        parser.add_argument('--beta',type=float,default=0.0)
        parser.add_argument('--using-gold-sememe',default=False)
        parser.add_argument('--y-encoder-type')
        parser.add_argument('--sememe-embed-path',default='./checkpoints/sememe_predictor/linear.sememe_vector')

    @classmethod
    def build_encoder(cls, task, args, dictionary):
        args.sememe_embed = load_pretrained_embedding_from_file(args.sememe_embed_path,args.sememe_dict,args.encoder_embed_dim)
        encoder = DVDGEncoder(
            src_dictionary=dictionary,
            char_embed=args.char_embed if hasattr(args,'char_embed') else None,
            latent_K=args.latent_K if not args.using_gold_sememe else len(args.sememe_dict),
            latent_M=args.latent_M if not args.using_gold_sememe else 0,
            pretrained_word_embed=args.pretrained_word_embed,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_x_encoder_embed=args.pretrained_encoder_embed,
            using_gold_sememe=args.using_gold_sememe,
            using_bert=args.using_bert,
            pretrained_sememe_embed=args.sememe_embed
        )
        return encoder

    @classmethod
    def build_decoder(
            cls, task, args, dictionary,encoder
    ):
        decoder = CategoricalDecoder(
            latent_M=args.latent_M if not args.using_gold_sememe else 0,
            latent_K=args.latent_K if not args.using_gold_sememe else len(args.sememe_dict),
            dictionary=dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_output_units=encoder.x_encoder.output_units,
            pretrained_embed=args.pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == "adaptive_loss"
                else None
            ),
            rnn_type=args.rnn_type,
            info_type=args.info_type,
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
        bert_repre=None,
        bag_of_sememe=None,
    ):
        encoder_out = self.encoder(
            src_tokens, src_lengths, word_tokens,bert_repre=bert_repre,bag_of_sememe=bag_of_sememe
        )
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out)
        return decoder_out

    def get_posterior_loss(self, net_output,sample,distance_function='BOW'):
        if self.using_gold_sememe:
            return torch.tensor(0.).cuda()
        target_tokens = sample['net_input']['target']
        target_lengths = sample['net_input']['target_lengths']
        word_tokens = sample['net_input']['word_tokens']
        if self.args.y_encoder_type in ['SIF','bow','unigram-bow']:
            target_emb = self.decoder.embed_tokens(target_tokens).detach()
            y_encoding = self.y_encoder(target_tokens,target_emb,target_lengths)
        elif self.args.y_encoder_type == 'BILSTM':
            y_encoding = self.y_encoder(target_tokens,target_lengths,word_tokens)
        latent_context = net_output[1]['latent_context']  # bsize * latent_M * D

        ## we want the bag of latent variables to paraphrase the target (definition or sememes)
        # either using average pooling or max pooling
        summary_of_latent = latent_context.sum(1)  # bsize * D
        if distance_function == 'cosine':
            reconstruction_loss = 1 - torch.nn.functional.cosine_similarity(summary_of_latent,y_encoding)
        elif distance_function == 'l2':
            reconstruction_loss = torch.pow(summary_of_latent - y_encoding,2).sum(-1)

        elif distance_function == 'BOW':
            if self.args.target_type == 'definition':
                bag_of_word = sample['bow']
                logits = F.linear(summary_of_latent,self.decoder.embed_tokens.weight.detach())
                prob = F.log_softmax(logits)
                target_mask = ~bag_of_word.eq(self.decoder.padding_idx)
                gathered_prob = torch.gather(prob,dim=-1,index=bag_of_word) * target_mask.float()
                reconstruction_loss = - gathered_prob.mean(-1)
            elif self.args.target_type == 'sememe':
                bag_of_sememe = sample['net_input']['bag_of_sememe']
                logits = F.linear(summary_of_latent,self.sememe_embed.weight)
                sememe_target = torch.zeros_like(logits).scatter_(-1,bag_of_sememe,1).float()
                sememe_target[:,self.decoder.padding_idx] = 0
                reconstruction_loss = F.binary_cross_entropy_with_logits(
                    logits.reshape(-1), sememe_target.reshape(-1), reduction="mean"
                )

        ## disagree loss
        pdist_mt = pairwise_cosine_similarity(latent_context)  # bsize * M * M
        disagreement_loss = pdist_mt.sum(-1).sum(-1) / (self.args.latent_M * self.args.latent_M)

        return reconstruction_loss + self.args.beta * disagreement_loss

    def get_z_code(self,sample):
        net_input = sample['net_input']
        with torch.no_grad():
            enc_x = self.encoder.x_encoder(net_input['src_tokens'], net_input['src_lengths'], net_input['word_tokens'],bert_repre=net_input['bert_repre'])
            z_code = self.encoder.compute_z_code(enc_x,bert_repre=net_input['bert_repre'])  # B * M * K
            # z_code = z_code.argmax(dim=-1)
        return z_code

    # def get_predicted_bow(self,sample):
    #     net_input = sample['net_input']
    #     with torch.no_grad():
    #         encoder_out = self.encoder(net_input['src_tokens'], net_input['src_lengths'],net_input['word_tokens'],bert_repre=net_input['bert_repre'])
    #         latent_context = encoder_out['latent_context']
    #         sum_of_latent = latent_context.sum(1)
    #         logits = torch.mm(sum_of_latent,self.decoder.embed_tokens.weight.t().detach())
    #     return (torch.sigmoid(logits) > 0.5).long()


@register_model_architecture("DVDG", "DVDG")
def base_architecture(args):
    set_default_args(args)
    args.latent_M = getattr(args, "latent_M", 8)
    args.latent_K = getattr(args,"latent_K",256)
    args.beta = getattr(args,'beta',0.0)
    args.y_encoder_type = 'unigram-bow'
    args.info_type = 'context,word,char,latent'
    args.target_type = 'sememe'


@register_model_architecture('DVDG','DVDG-unigram-bow')
def dvdg_unigram_bow(args):
    base_architecture(args)
    args.y_encoder_type = 'unigram-bow'
    args.target_type = 'definition'


@register_model_architecture('DVDG','DVDG-SIF')
def dvdg_sif(args):
    base_architecture(args)
    args.y_encoder_type = 'SIF'
    args.target_type = 'definition'


@register_model_architecture('DVDG','DVDG-BOW')
def dvdg_bow(args):
    base_architecture(args)
    args.y_encoder_type = 'BOW'
    args.target_type = 'definition'
    args.seed = random.randint(0,100)

@register_model_architecture('DVDG','DVDG-Bottleneck')
def dvdg_bottleneck(args):
    base_architecture(args)
    args.y_encoder_type = 'BOW'
    args.target_type = 'definition'
    args.info_type = 'context,latent,char'


@register_model_architecture('DVDG','BertDVDG-bow')
def dvdg_bert_bow(args):
    dvdg_bow(args)
    args.info_type = 'context,word,char,bert'
    args.using_bert = True

@register_model_architecture('DVDG','DVDG-GoldSememe')
def dvdg_gold_sememe(args):
    base_architecture(args)
    args.using_gold_sememe = True

@register_model_architecture('DVDG','DVDG-Sememe')
def dvdg_sememe(args):
    base_architecture(args)
    args.target_type = 'sememe'
    args.y_encoder_type = 'BOW'

@register_model_architecture('DVDG','DVDG-KL-BOW')
def dvdg_kl_bow(args):
    base_architecture(args)
    args.y_encoder_type = 'BOW'
    args.target_type = 'definition'
