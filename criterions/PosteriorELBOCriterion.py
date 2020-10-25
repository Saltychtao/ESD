# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn.functional as F
import pdb

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss


@register_criterion('PosteriorELBO')
class PosteriorELBOCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        nll_loss, posterior_loss = self.compute_loss(model, net_output, sample)
        loss = nll_loss + posterior_loss.sum()
        sample_size = sample['ntokens']
        nsentences = sample['target'].size()[0]
        logging_output = {
            'nll_loss': nll_loss.item(),
            'posterior_loss': posterior_loss.sum().item(),
            'loss': loss.data.item(),
            'ntokens': sample['ntokens'],
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return nll_loss,posterior_loss,sample_size, logging_output

    def compute_loss(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            epsilon=0.1,
            ignore_index=self.padding_idx,
            reduce=True
        )
        posterior_loss = model.get_posterior_loss(net_output,sample)

        return loss,posterior_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        posterior_loss_sum = sum(log.get('posterior_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / nsentences if nsentences > 0 else 0.,
            'nll_loss': nll_loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'posterior_loss': posterior_loss_sum / nsentences,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = nll_loss_sum / ntokens / math.log(2)
        return agg_output
