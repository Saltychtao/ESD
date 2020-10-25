# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn.functional as F

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion
import pdb


@register_criterion('two_pass_cross_entropy')
class TwoPassCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # pdb.set_trace()
        net_output = model(**sample['net_input'])
        first_pass_loss, second_pass_loss = self.compute_loss(model, net_output, sample,)
        loss = first_pass_loss + second_pass_loss
        sample_size = sample['target1'].size(0) + sample['target2'] if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'loss_1': utils.item(first_pass_loss.data) if reduce else loss.data,
            'loss_2': utils.item(second_pass_loss.data) if reduce else loss.data,
            'ntokens': 2*sample['ntokens'],
            'nsentences': sample['target1'].size(0),
            'sample_size': sample_size,
        }
        return first_pass_loss,second_pass_loss,sample_size, logging_output

    def compute_loss(self, model, net_output, sample,):
        first_pass_output, second_pass_output = net_output

        ## in first pass, loss of all the tokens are accounted
        ## pass (first_pass_x,net_output) into the get_normalized_probs
        lprobs = model.get_normalized_probs(first_pass_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        first_pass_target = model.get_first_pass_targets(sample, net_output).view(-1)
        first_pass_loss = F.nll_loss(
            lprobs,
            first_pass_target,
            ignore_index=self.padding_idx,
            reduction='none'
        )

        ## in second pass,  loss of those which doesnt appear in the first pass are accounted
        lprobs = model.get_normalized_probs(second_pass_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        second_pass_target = model.get_second_pass_targets(sample, net_output).view(-1)
        second_pass_loss = F.nll_loss(
            lprobs,
            second_pass_target,
            ignore_index=self.padding_idx,
            reduction='none'
        )
        # pdb.set_trace()
        # second_pass_decoder_mask = ~first_pass_target.eq(second_pass_target).byte()
        # second_pass_loss = second_pass_loss * second_pass_decoder_mask

        # loss = first_pass_loss.sum() + second_pass_loss.sum()

        return first_pass_loss.sum(), second_pass_loss.sum()

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
