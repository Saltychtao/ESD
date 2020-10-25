# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn.functional as F

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('ELBO')
class ELBOCriterion(FairseqCriterion):

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
        nll_loss, kld_loss = self.compute_loss(model, net_output, sample)
        ELBO = nll_loss + kld_loss.sum()
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        nsentences = sample['target'].size()[0]
        logging_output = {
            'nll_loss': nll_loss.item(),
            'kld_loss': kld_loss.sum().item(),
            'loss': ELBO.data.item(),
            'ntokens': sample['ntokens'],
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return nll_loss,kld_loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        nll_loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            size_average=False
        )
        kld_loss = model.get_kld_loss(net_output)
        return nll_loss, kld_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        elbo_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        kld_loss_sum = sum(log.get('kld_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': elbo_sum / nsentences if nsentences > 0 else 0.,
            'nll_loss': nll_loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'kld_loss': kld_loss_sum / nsentences if nsentences > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = nll_loss_sum / ntokens / math.log(2)
        return agg_output
