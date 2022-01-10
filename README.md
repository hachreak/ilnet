## Improving Localization for Semi-Supervised Object Detection

### Installation

See [unbiased-teacher](https://github.com/facebookresearch/unbiased-teacher/tree/ba543ed)
official pages to know the installation procedure.

### Models

Most important files.

- [trainer.py](ubteacher/engine/trainer.py)
- [fast_rcnn](ubteacher/modeling/roi_heads/fast_rcnn.py) which contains the
  BBox IoU classification branch.

## Experiments

The following configuration files have been used to run each experiment:

Table 1 / Figure 3a:

| Row | Beta | AP |
| :--: | :--: | :--: |
|   | [base](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-base.yaml)  | 31.027 |
| 0 | [0.5](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v7_4.yaml)   | 31.775 |
| 1 | [1](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v7.yaml)       | 31.947 |
| 2 | [2](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v7_2.yaml)     | 31.754 |
| 3 | [4](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v7_3.yaml)     | 30.445 |


Table 2 / Figure 3c:

| Row | Model | AP |
| :--: | :--: | :--: |
| 1 | [UT](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-base.yaml)                 | 31.027 |
| 2 | [Ours (with filter)](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v7_4.yaml) | 31.605 |
| 3 | [Ours (w/out filter)](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v7.yaml)  | 31.509 |

Table 3 / Figure 3d:

| Row | Mu | AP |
| :--: | :--: | :--: |
| 1 | [0.5](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v28_2.yaml)   | 31.199 |
| 2 | [0.6](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v28.yaml)     | 31.128 |
| 3 | [0.7](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v28_3.yaml)   | 31.461 |
| 4 | [0.75](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v13_5.yaml)  | 31.604 |
| 5 | [0.8](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v28_4.yaml)   | 31.336 |
| 6 | [0.9](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v28_5.yaml)   | 27.125 |

Table 4:

| Row | Theta | AP |
| :--: | :--: | :--: |
| 0 | [0.3](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v31_4.yaml)  | 31.404 |
| 1 | [0.4](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v31_3.yaml)  | 31.630 |
| 2 | [0.5](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v13_5.yaml)  | 31.604 |
| 3 | [0.6](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v31.yaml)    | 31.158 |
| 4 | [0.7](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v31_2.yaml)  | 30.649 |

Table 5:

| Row | L<sub>reg</sub><sup>unsup</sup> | x<sup>sh</sup> | scores | deltas| Model | AP |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 1 |   |   |   |   | [model](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-base.yaml)   | 31.027 |
| 2 | x |   |   |   | [model](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v7.yaml)     | 31.947 |
| 3 | x | x |   |   | [model](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v36_3.yaml)  | 31.754 |
| 4 | x | x | x |   | [model](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v36_2.yaml)  | 32.166 |
| 5 | x | x | x | x | [model](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v36.yaml)    | 31.923 |
| 6 |   | x | x | x | [model](configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-v31_3.yaml)  | 31.630 |
