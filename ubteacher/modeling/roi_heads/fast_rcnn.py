# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List
from detectron2.structures import Boxes, Instances
from detectron2.layers import nonzero_tuple

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    fast_rcnn_inference
)

try:
    from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
except ImportError:
    from .backports import FastRCNNOutputs


# focal loss
class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        # config bbox iou branch
        self.with_iou_pred = cfg.SEMISUPNET.WITH_IOU_PRED
        self.iou_threshold = cfg.SEMISUPNET.IOU_PRED_THRESHOLD
        self.iou_inference_threshold = cfg.SEMISUPNET.IOU_PRED_INFERENCE_THRESHOLD
        self.filter_with_bbox_iou = cfg.SEMISUPNET.FILTER_WITH_BBOX_IOU
        self.with_score_values = cfg.SEMISUPNET.WITH_SCORE_VALUES
        self.with_iou_pred_elu = cfg.SEMISUPNET.WITH_IOU_PRED_ELU
        self.with_regr_values = cfg.SEMISUPNET.WITH_REGR_VALUES
        self.with_bbox_iou_regres = cfg.SEMISUPNET.WITH_BBOX_IOU_REGRES
        self.use_smoothl1 = cfg.SEMISUPNET.USE_SMOOTHL1

        if self.with_iou_pred:
            inter_channels = input_shape.channels
            in_channels = inter_channels

            if self.with_score_values:
                in_channels += (self.num_classes + 1)

            if self.with_regr_values:
                in_channels += (self.num_classes * 4)

            if not self.with_iou_pred_elu:
                self.iou_pred = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, inter_channels),
                    torch.nn.Linear(inter_channels, self.num_classes + 1)
                )
            else:
                self.iou_pred = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, inter_channels),
                    nn.ELU(),
                    torch.nn.Linear(inter_channels, self.num_classes + 1)
                )

            def init_bbox_iou_weights(m):
                if type(m) == nn.Linear:
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.constant_(m.bias, 0)

            self.iou_pred.apply(init_bbox_iou_weights)

    def forward(self, x):
        scores, proposal_deltas = super(FastRCNNFocaltLossOutputLayers, self).forward(x)

        iou_scores = None
        if self.with_iou_pred:
            deltas = proposal_deltas

            iou_input = x
            if self.with_score_values:
                iou_input = torch.cat([iou_input, scores], dim=1)
            if self.with_regr_values:
                iou_input = torch.cat([iou_input, deltas], dim=1)

            iou_scores = self.iou_pred(iou_input)

        return scores, proposal_deltas, iou_scores

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas, iou_scores = predictions

        losses = FastRCNNFocalLoss(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
            iou_threshold=self.iou_threshold,
            iou_scores=iou_scores,
            with_bbox_iou_regres=self.with_bbox_iou_regres,
            use_smoothl1=self.use_smoothl1,
        ).losses()

        return losses

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances], branch: str = ""):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        logger = logging.getLogger('fvcore.common.checkpoint')
        boxes = self.predict_boxes(predictions[:2], proposals)
        scores = self.predict_probs(predictions[:2], proposals)
        # filter by iou score
        if len(predictions) > 2 and predictions[2] is not None:
            iou_scores = self.predict_iou(predictions[-1], proposals)
            if self.filter_with_bbox_iou:
                clss = []
                for s in scores:
                    clss.append(s.max(1)[1] if len(s) > 0 else torch.tensor([], dtype=torch.long, device=s.device))
                counts = [b.shape[0] for b in boxes]
                logger.info("bbox_iou bg {}".format(sum([len(s[s==self.num_classes]) for s in clss])))
                boxes = self.filter(boxes, iou_scores, clss)
                logger.info("bbox_iou filtering {}".format(sum([c-b.shape[0] for (c, b) in zip(counts, boxes)])))
                scores = self.filter(scores, iou_scores, clss)
                iou_scores = self.filter(iou_scores, iou_scores, clss)

        image_shapes = [x.image_size for x in proposals]

        # only on teacher pseudo-labeling!
        if branch == 'unsup_data_weak' and self.with_teacher_filter_preds:
            th = self.teacher_filter_preds_min_delta
            boxes, scores = self._filter_preds(boxes, scores, th)

        instances, idxs = fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

        if len(predictions) > 2 and predictions[2] is not None:
            instances = self.add_iou_scores(iou_scores, instances, idxs)

        return instances, idxs

    def _filter_preds(self, boxes, scores, min_delta):
        boxes = list(boxes)
        scores = list(scores)
        for idx in range(len(boxes)):
            top2 = scores[idx].topk(2)[0]
            filter_ = (top2[:, 0] - top2[:, 1]) > min_delta
            boxes[idx] = boxes[idx][filter_]
            scores[idx] = scores[idx][filter_]
        return boxes, scores

    def predict_iou(self, preds, proposals):
        num_inst_per_image = [len(p) for p in proposals]
        preds = preds.sigmoid()
        return preds.split(num_inst_per_image, dim=0)

    def add_iou_scores(self, iou_scores, instances, idxs):
        for i, instance in enumerate(instances):
            ious = iou_scores[i][idxs[i]]
            counter = torch.arange(0, len(instance))
            ious = ious[counter, instance.pred_classes]
            instance.pred_iou_scores = ious
        return instances

    def filter(self, x, iou_scores, scores):
        res = []
        for idx, val in enumerate(x):
            ious = iou_scores[idx][torch.arange(0, scores[idx].shape[0]), scores[idx]]
            fit = (
                (scores[idx] < self.num_classes) & (ious > self.iou_inference_threshold)
            ) | (scores[idx] == self.num_classes)
            res.append(val[fit])
        return tuple(res)


class FastRCNNFocalLoss(FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        num_classes=80,
        iou_threshold=0.7,
        iou_scores=None,
        with_bbox_iou_regres=False,
        use_smoothl1=False,
    ):
        super(FastRCNNFocalLoss, self).__init__(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
            box_reg_loss_type,
        )
        self.num_classes = num_classes

        # bbox iou branch
        self.iou_threshold = iou_threshold
        self.iou_scores = iou_scores
        self.with_bbox_iou_regres = with_bbox_iou_regres
        self.use_smoothl1 = use_smoothl1

    def losses(self):
        losses = {
            "loss_cls": self.comput_focal_loss(),
            "loss_box_reg": self.box_reg_loss(),
        }

        # bbox iou branch
        if self.iou_scores is not None:
            loss_box_iou = self.box_iou_loss(self.gt_classes)
            losses["loss_box_iou"] = loss_box_iou

        return losses

    def comput_focal_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            FC_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )
            total_loss = FC_loss(input=self.pred_class_logits, target=self.gt_classes)
            total_loss = total_loss / self.gt_classes.shape[0]
            return total_loss

    def build_bboxes_iou(self, gt_classes):
        box_dim = self.proposals.tensor.size(1)  # 4 or 5
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds should produce a valid loss of zero because reduction=sum.
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < bg_class_ind))[0]

        if len(fg_inds) == 0:
            return None, None

        # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
        # where b is the dimension of box representation (4 or 5)
        # Note that compared to Detectron1,
        # we do not perform bounding box regression for background classes.
        gt_class_cols = box_dim * gt_classes[fg_inds, None] + torch.arange(
            box_dim, device=device
        )

        fg_pred_boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            self.proposals.tensor[fg_inds],
        )

        return get_iou(fg_pred_boxes, self.gt_boxes.tensor[fg_inds]), fg_inds

    def box_iou_loss(self, gt_classes):
        if self._no_instances or self.iou_scores is None:
            return 0.0 * self.pred_proposal_deltas.sum()

        iou_level, fg_inds = self.build_bboxes_iou(gt_classes)
        if iou_level is None:
            return 0.0 * self.pred_proposal_deltas.sum()

        if not self.use_smoothl1:
            loss = FocalLoss(gamma=1.5, num_classes=1)
        else:
            loss = nn.SmoothL1Loss(reduction='sum')

        logger = logging.getLogger('fvcore.common.checkpoint')
        logger.info('bbox_iou loss tot_pos {} {}'.format(len(iou_level), len(iou_level[(iou_level > self.iou_threshold)])))

        if self.with_bbox_iou_regres:
            m = nn.Sigmoid()
            loss_value = loss(m(self.iou_scores[fg_inds, gt_classes[fg_inds]]), iou_level.detach())
            if self.use_smoothl1:
                return loss_value / max(iou_level.numel(), 1.)
        else:
            # sigmoid
            m = nn.Sigmoid()
            loss_value = loss(m(self.iou_scores[fg_inds, gt_classes[fg_inds]]), (iou_level > self.iou_threshold).float())

        return loss_value / max(gt_classes.numel(), 1.0)


def get_iou(boxes1, boxes2):
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    return intsctk / (unionk + 1e-7)

class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
        reduction='sum'
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, input, target):
        # focal loss
        if self.num_classes > 1:
            CE = F.cross_entropy(input, target, reduction="none")
        else:
            CE = F.binary_cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        if self.reduction == 'none':
            return loss
        return loss.sum()

