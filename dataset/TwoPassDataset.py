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
    input_feeding=True,
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

    prev_output_tokens1 = None
    target1 = None
    target1_lengths = None

    prev_output_tokens2 = None
    target2 = None
    target2_lengths = None
    if samples[0].get('target1', None) is not None:
        target1 = merge('target1', left_pad=left_pad_target)
        target1 = target1.index_select(0, sort_order)
        target1_lengths = torch.LongTensor([s['target1'].numel() for s in samples])

        target2 = merge('target2', left_pad=left_pad_target)
        target2 = target2.index_select(0, sort_order)
        target2_lengths = torch.LongTensor([s['target2'].numel() for s in samples])

        ntokens = sum(len(s['target2']) for s in samples)
        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens1 = merge(
                'target1',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens1 = prev_output_tokens1.index_select(0, sort_order)

            prev_output_tokens2 = merge(
                'target2',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens2 = prev_output_tokens2.index_select(0, sort_order)
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
            'target1': target1,
            'target1_lengths': target1_lengths,
        },
        'target1': target1,
        'target1_lengths': target1_lengths,
        'target': target2,
        'target_lengths': target2_lengths
    }
    if prev_output_tokens1 is not None:
        batch['net_input']['prev_output_tokens1'] = prev_output_tokens1
        batch['net_input']['prev_output_tokens2'] = prev_output_tokens2
    return batch


class TwoPassDataset(FairseqDataset):
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
            self, words,src, src_sizes,
            src_dict,tgt_dict,
            tgt1=None, tgt1_sizes=None,
            tgt2=None, tgt2_sizes=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.words = words
        self.src = src
        self.tgt1 = tgt1
        self.tgt2 = tgt2
        self.src_sizes = np.array(src_sizes)
        self.tgt1_sizes = np.array(tgt1_sizes) if tgt1_sizes is not None else None
        self.tgt2_sizes = np.array(tgt2_sizes) if tgt2_sizes is not None else None
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
        tgt1_item = self.tgt1[index] if self.tgt1 is not None else None
        tgt2_item = self.tgt2[index] if self.tgt2 is not None else None
        src_item = self.src[index]
        word_item = self.words[index][0]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt1_dict else self.src1_dict.eos()
            if self.tgt1 and self.tgt1[index][-1] != eos:
                tgt1_item = torch.cat([self.tgt1[index], torch.LongTensor([eos])])
                tgt2_item = torch.cat([self.tgt2[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        return {
            'id': index,
            'source': src_item,
            'target1': tgt1_item,
            'target2': tgt2_item,
            'word': word_item
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
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt2_sizes[index] if self.tgt2_sizes is not None else 0) + 1

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt2_sizes[index] if self.tgt2_sizes is not None else 0)

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
            and (getattr(self.tgt1, 'supports_prefetch',False) or self.tgt1 is None)
            and (getattr(self.tgt2, 'supports_prefetch',False) or self.tgt2 is None)
            and (getattr(self.words,'supports_prefetch',False))
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt1 is not None:
            self.tgt1.prefetch(indices)
        if self.tgt2 is not None:
            self.tgt2.prefetch(indices)
        self.words.prefetch(indices)
