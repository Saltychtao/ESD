# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import pdb

from fairseq.data import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,target_size=None
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    word_tokens = torch.LongTensor([s['word'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    word_tokens = word_tokens.index_select(0,sort_order)

    if samples[0].get('bert_encoding',None) is not None:
        bert_repre = torch.stack([s['bert_encoding'] for s in samples],dim=0)
        bert_repre = bert_repre.index_select(0,sort_order)
    else:
        bert_repre = None

    if samples[0].get('bag_of_clean_sememe',None) is not None and samples[0].get('clean_words',None) is not None:
        bag_of_clean_sememe = merge('bag_of_clean_sememe',left_pad=left_pad_target)
        clean_word_tokens = torch.LongTensor([s['clean_words'] for s in samples])
    else:
        bag_of_clean_sememe = None
        clean_word_tokens = None

    prev_output_tokens = None
    target = None
    target_lengths = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        # ppmi = merge('ppmi',left_pad=left_pad_target)
        # ppmi = ppmi.index_select(0,sort_order)
        bow = merge('bow',left_pad=left_pad_target)
        bow = bow.index_select(0,sort_order)
        # nbow = torch.FloatTensor(bow.size()[0],20).uniform_(2,target_size-1).long()
        ntokens = sum(len(s['target']) for s in samples)
        target_lengths = torch.LongTensor([s['target'].numel() for s in samples])
        target_lengths = target_lengths.index_select(0,sort_order)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'word_tokens': word_tokens,
            'bert_repre': bert_repre,
            'target': target,
            'target_lengths': target_lengths
        },
        'target': target,
        'target_lengths': target_lengths,
        # 'ppmi': ppmi,
        'bow': bow,
        'bag_of_clean_sememe': bag_of_clean_sememe,
        'clean_word_tokens': clean_word_tokens
        # 'nbow': nbow
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class SememeDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
            self, words,
            src,src_sizes, src_dict,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
            ppmi_map=None,bert_repre=None,clean_sememes=None,clean_words=None
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.clean_sememes = clean_sememes
        self.clean_words = clean_words
        self.bert_repre = bert_repre
        self.ppmi_map = ppmi_map
        self.words = words
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        word_item = self.words[index][0]

        # ppmi_item = []
        # for c in tgt_item:
        #     ppmi_item.append(self.ppmi_map.get((word_item.item(),c.item()),0))
        # ppmi_item = torch.FloatTensor(ppmi_item)
        bow_item = set()
        for c in tgt_item:
            if (word_item.item(),c.item()) in self.ppmi_map:
                bow_item.add(c.item())
        # bow_item = torch.zeros(len(self.tgt_dict)).scatter_(0,torch.tensor(list(bow_item)).long(),1)
        bow_item = torch.LongTensor(list(bow_item))

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.bert_repre is not None:
            bert_encoding = self.bert_repre[index]
        else:
            bert_encoding = None

        # if self.sememes is not None:
        #     bag_of_sememe = self.sememes[index]
        #     eos = self.src_dict.eos()
        #     if bag_of_sememe[-1] == eos:
        #         bag_of_sememe = bag_of_sememe[:-1]
        # else:
        #     bag_of_sememe = None

        if self.clean_sememes is not None:
            bag_of_clean_sememe = self.clean_sememes[index % len(self.clean_sememes)]
            clean_words = self.clean_words[index % len(self.clean_words)][0]
            eos = self.src_dict.eos()
            if bag_of_clean_sememe[-1] == eos:
                bag_of_clean_sememe = bag_of_clean_sememe[:-1]
        else:
            bag_of_clean_sememe = None
            clean_words = None

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'word': word_item,
            # 'ppmi': ppmi_item,
            'bow': bow_item,
            'bert_encoding': bert_encoding,
            'clean_words': clean_words,
            'bag_of_clean_sememe': bag_of_clean_sememe
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,target_size=len(self.tgt_dict)
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0) + 1

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    # def ordered_indices(self):
    #     """Return an ordered list of indices. Batches will be constructed based
    #     on this order."""
    #     if self.shuffle:
    #         indices = np.random.permutation(len(self))
    #     else:
    #         indices = np.arange(len(self))
    #     if self.tgt_sizes is not None:
    #         indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
    #     return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
            and (getattr(self.words,'supports_prefetch',False))
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        self.words.prefetch(indices)
