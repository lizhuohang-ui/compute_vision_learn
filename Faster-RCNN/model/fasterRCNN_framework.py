import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from model.roi_head import RoIHeads
from model.transform import GeneralizedRCNNTransform
from model.rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork


class FasterRCNN_Base(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transform):
        super(FasterRCNN_Base, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self._has_warned = False

    @torch.jit.unused
    def eager_output(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training and targets is None:
            raise ValueError("In training mode, target should be passed")

        if self.training:
            assert targets is not None
            for target in targets: # 判断target中bbox是否符合要求
                boxes = target['boxes']
                if isinstance(boxes, Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         f"of shape [N, 4], got{boxes.shape[:]}")
                else:
                    raise ValueError('Expected target boxes to be a tensor'
                                     f'but got{type(boxes)}')

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for image in images:
            val = image.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))  # 原始图片的高宽

        images, targets = self.transform(images, targets)

        # 将图片输入backbone得到feature map
        feature_map = self.backbone(images.tensors)
        if isinstance(feature_map, torch.Tensor):
            feature_map = OrderedDict([('0', feature_map)])

        # 将feature map和target信息传给RPN
        # proposals: List[Tensor], Tensor.shape: [num_proposals, 4]  (x1, y1, x2, y2)
        proposals, proposal_loss = self.rpn(images, feature_map, targets)

        # 将RPN生成的数据和target信息传递给ROIPool层
        detections, detector_loss = self.roi_heads(feature_map, proposals, images.image_sizes, targets)

        # postprocess（torch）： 对网络的预测结果进行后处理（主要是将bbox还原到原始图片上）
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)


        losses = {}
        losses.update(detector_loss)
        losses.update(proposal_loss)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (losses, detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_output(losses, detections)



class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    """
    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class FasterRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Faster R-CNN.
    """
    def __init__(self, in_channels, num_classes):
        super(FasterRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return scores, bbox_pred


class FasterRCNN(FasterRCNN_Base):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # rpn计算损失时采样的样本数，以及正样本占总样本的比例
                 rpn_score_thresh=0.0,
                 # BBox parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detection_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 ):
        if not hasattr(backbone, 'out_channels'):
            raise ValueError(
                "backbone should contain an attribute out_channels"
                "specifying the number of output channels  (assumed to be the"
                "same for all the levels)"
            )

        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )

        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=[7, 7],
                sampling_ratio=2,
            )

        if box_head is None:
            resolution = box_roi_pool.output_size[0]  # default=7
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size,
            )
        
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FasterRCNNPredictor(
                representation_size,
                num_classes,
            )
        
        roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detection_per_img,
        )
        
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        
        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)

# if __name__ == "__main__":
#     x = {'A': 1}
#     print(x)




