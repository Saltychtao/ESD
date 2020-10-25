# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import numpy as np
import torch
import pdb

from fairseq import options, utils
from fairseq import tokenizer
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
    Dictionary
)

from VariationalDefinitionGeneration.dataset.VDataset import VDGMDataset

from fairseq.tasks import FairseqTask
import fairseq


def load_langpair_dataset(
        data_path, split,
        src, src_dict,
        tgt, tgt_dict,
        combine, dataset_impl, upsample_primary,
        left_pad_source, left_pad_target, max_source_positions, max_target_positions,
        ppmi_map=None, using_bert=False,using_sememe=True,sememe_dict=None
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_datasets.append(
            data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        )
        tgt_datasets.append(
            data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        )

        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    words = data_utils.load_indexed_dataset(os.path.join(data_path,'{}.word'.format(split)),src_dict)
    if words is None:
        print(' Loaded Words Failed!')

    if using_bert:
        bert_repre = torch.load(os.path.join(data_path,'{}.br.pth'.format(split)),map_location=torch.device('cpu'))
        print(' Loaded Pretrained Bert Representation of {} split!'.format(split))
    else:
        bert_repre = None

    if using_sememe:
        sememes = data_utils.load_indexed_dataset(os.path.join(data_path,'{}.sememe').format(split),sememe_dict)
    else:
        sememes = None

    return VDGMDataset(
        words,
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        ppmi_map=ppmi_map,
        bert_repre=bert_repre,
        sememes=sememes
    )


@fairseq.tasks.register_task('VDGM')
class VDGMTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--max-src-vocab-size',default=50000,type=int)
        parser.add_argument('--max-tgt-vocab-size',default=50000,type=int)
        parser.add_argument('--vocab-min-count',default=1,type=int)
        parser.add_argument('--kl-weight-k',type=float,default=0.005)
        parser.add_argument('--kl-weight-x0',type=int,default=1200)
        parser.add_argument('--anneal-function',default='logistic')
        parser.add_argument('--kl-freebits',default=0.0,type=float)
        parser.add_argument('--alpha',default=0.1,type=float)
        parser.add_argument('--using-bert',metavar='BOOL',default=False)
        parser.add_argument('--using-sememe',metavar='BOOL',default=False)

        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, sememe_dict=None):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.sememe_dict = sememe_dict
        self.alpha = args.alpha
        self.using_bert = args.using_bert
        self.using_sememe = True
        self.num_of_updates = 0

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))

        # optionally build sememe dictionary
        sememe_dict = Dictionary()
        Dictionary.add_file_to_dictionary(os.path.join(paths[0],'train.sememe'),sememe_dict,tokenizer.tokenize_line, num_workers=12)
        args.sememe_dict = sememe_dict

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        print('| [{}] dictionary: {} types'.format('sememe', len(sememe_dict)))

        return cls(args, src_dict, tgt_dict, sememe_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang
        # print('Loading PPMI...')
        # ppmi_map = dict()
        # with open(os.path.join(data_path,'{}.cooccur'.format('train')),'r') as f:
        #     for line in f:
        #         w,c,ppmi = tuple(line.split())
        #         w = self.src_dict.index(w)
        #         c = self.tgt_dict.index(c)
        #         ppmi_map[(w,c)] = float(ppmi)

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            ppmi_map=None,
            using_bert=self.using_bert,
            using_sememe=True,
            sememe_dict=self.sememe_dict
        )

    def build_dataset_for_inference(self,positions,words,src_tokens, src_lengths):
        return VDGMDataset(positions,words,src_tokens, src_lengths, self.source_dictionary)

    def build_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq.sequence_generator import SequenceGenerator
            return SequenceGenerator(
                self.target_dictionary,
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),
                beam_size=getattr(args, 'beam', 5),
                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def update_step(self,num_of_updates):
        self.num_of_updates = num_of_updates

    def train_step(self,sample,model,criterion,optimizer,ignore_gard=False):
        model.train()
        ## TODO define proper forward and backward
        nll_loss,posterior_loss,sample_size, logging_output = criterion(model,sample,criterion)
        loss = (nll_loss + self.alpha * posterior_loss.sum())
        # else:
        #     loss = posterior_loss.sum()
        if ignore_gard:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self,sample,model,criterion):
        model.eval()
        with torch.no_grad():
            nll_loss,posterior_loss,sample_size,logging_output = criterion(model,sample,criterion)
            elbo = nll_loss
        return elbo,sample_size,logging_output

    def aggregate_logging_outputs(self,logging_outputs,criterion):
        logging_output = criterion.__class__.aggregate_logging_outputs(logging_outputs)
        return logging_output

    def inference_step(self,generator,models,sample,prefix_tokens=None):
        with torch.no_grad():
            hypos = generator.generate(models,sample,prefix_tokens=prefix_tokens)
        # z_code = models[0].get_z_code(sample)  # bsize * M
        # return hypos,z_code
        return hypos