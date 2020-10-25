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
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
    Dictionary
)

from VariationalDefinitionGeneration.dataset.TwoPassDataset import TwoPassDataset

from fairseq.tasks import FairseqTask
import fairseq


def load_langpair_dataset(
        data_path, split,
        src, src_dict,
        tgt_dict,
        tgt1,tgt2,
        combine, dataset_impl, upsample_primary,
        left_pad_source, left_pad_target, max_source_positions, max_target_positions,
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    tgt = "['tgt1', 'tgt2']"

    src_dataset = data_utils.load_indexed_dataset(os.path.join(data_path,'{}.{}-{}.{}'.format(split,src,tgt,src)),src_dict,dataset_impl)
    print('| {} {} {}-{} {} examples'.format(data_path, split, src, tgt, len(src_dataset)))

    tgt1_dataset = data_utils.load_indexed_dataset(os.path.join(data_path,'{}.{}-{}.{}'.format(split,src,tgt,tgt1)),tgt_dict,dataset_impl)
    print('| {} {} {}-{} {} examples'.format(data_path, split, src, tgt, len(tgt1_dataset)))

    tgt2_dataset = data_utils.load_indexed_dataset(os.path.join(data_path,'{}.{}-{}.{}'.format(split,src,tgt,tgt2)),tgt_dict,dataset_impl)
    print('| {} {} {}-{} {} examples'.format(data_path, split, src, tgt, len(tgt2_dataset)))

    words = data_utils.load_indexed_dataset(os.path.join(data_path,'{}.word'.format(split)),src_dict)
    if words is None:
        print(' Loaded Words Failed!')

    return TwoPassDataset(
        words,
        src_dataset, src_dataset.sizes,
        src_dict,tgt_dict,
        tgt1_dataset, tgt1_dataset.sizes,
        tgt2_dataset, tgt2_dataset.sizes,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
    )


@fairseq.tasks.register_task('TwoPass')
class TwoPassTask(FairseqTask):
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
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',nargs='+',
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
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.alpha = 0

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
        with open(os.path.join(paths[0],'words.txt')) as f:
            for line in f:
                src_dict.add_symbol(line.rstrip())
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang[1], len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt1, tgt2 = self.args.source_lang, self.args.target_lang[0], self.args.target_lang[1]

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src,
            self.src_dict,self.tgt_dict,
            tgt1, tgt2,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    def build_dataset_for_inference(self,positions,words,src_tokens, src_lengths):
        return TwoPassDataset(positions,words,src_tokens, src_lengths, self.source_dictionary)

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

    def build_generator(self,args):
        from fairseq.sequence_generator import SequenceGenerator
        first_pass_generator = SequenceGenerator(
            self.target_dictionary,
            beam_size=getattr(args, 'beam', 5),
            max_len_a=getattr(args, 'max_len_a', 0),
            max_len_b=getattr(args, 'max_len_b', 200),
            min_len=getattr(args, 'min_len', 1),
            normalize_scores=(not getattr(args, 'unnormalized', False)),
            len_penalty=getattr(args, 'lenpen', 1),
            unk_penalty=getattr(args, 'unkpen', 0),
            sampling=getattr(args, 'sampling', False),
            sampling_topk=getattr(args, 'sampling_topk', -1),
            sampling_topp=getattr(args, 'sampling_topp', -1.0),
            temperature=getattr(args, 'temperature', 1.),
            diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
            diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0))
        second_pass_generator = SequenceGenerator(
            self.target_dictionary,
            beam_size=getattr(args, 'beam', 5),
            max_len_a=getattr(args, 'max_len_a', 0),
            max_len_b=getattr(args, 'max_len_b', 200),
            min_len=getattr(args, 'min_len', 1),
            normalize_scores=(not getattr(args, 'unnormalized', False)),
            len_penalty=getattr(args, 'lenpen', 1),
            unk_penalty=getattr(args, 'unkpen', 0),
            sampling=getattr(args, 'sampling', False),
            sampling_topk=getattr(args, 'sampling_topk', -1),
            sampling_topp=getattr(args, 'sampling_topp', -1.0),
            temperature=getattr(args, 'temperature', 1.),
            diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
            diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0))

        return [first_pass_generator,second_pass_generator]

    def update_step(self,num_updates):
        self.alpha = float(1 / (1 + np.exp(-0.0025 *(num_updates - 1000))))

    def train_step(self,sample,model,criterion,optimizer,ignore_gard=False):
        model.train()
        ## TODO define proper forward and backward
        loss_1,loss_2,sample_size, logging_output = criterion(model,sample,criterion)
        loss = loss_1 + self.alpha * loss_2
        if ignore_gard:
            loss *= 0
        optimizer.backward(loss)
        # optimizer.backward(posterior_loss.sum())
        return loss, sample_size, logging_output

    def valid_step(self,sample,model,criterion):
        model.eval()
        with torch.no_grad():
            loss_1,loss_2, sample_size, logging_output = criterion(model,sample,criterion)
            loss = loss_1 + self.alpha * loss_2
        return loss,sample_size,logging_output

    def aggregate_logging_outputs(self,logging_outputs,criterion):
        logging_output = criterion.__class__.aggregate_logging_outputs(logging_outputs)
        return logging_output

    def inference_step(self,generater,models,sample,prefix_tokens=None):

        def merge(samples,key, left_pad, move_eos_to_beginning=False):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                self.tgt_dict.pad(), self.tgt_dict.eos(), left_pad, move_eos_to_beginning,
        )
        with torch.no_grad():
            first_pass_hypos = generater[0].generate([models[0].first_pass_model],sample,prefix_tokens=prefix_tokens)
            first_pass_samples = [s[0] for s in first_pass_hypos]
            first_pass_tokens = merge(first_pass_samples,'tokens',left_pad=False)
            first_pass_tokens_lengths = torch.LongTensor([s['tokens'].numel() for s in first_pass_samples])
            second_pass_sample = {
                'net_input':{
                    'src_tokens': first_pass_tokens,
                    'src_lengths': first_pass_tokens_lengths,
                    'word_tokens': sample['net_input']['word_tokens'],
                }
            }

            second_pass_hypos = generater[1].generate([models[0].second_pass_model],second_pass_sample,prefix_tokens=prefix_tokens)
        return second_pass_hypos

    def aggregate_logging_outputs(self,logging_outputs,criterion):
        logging_output = criterion.__class__.aggregate_logging_outputs(logging_outputs)
        logging_output['alpha'] = self.alpha

        return logging_output
