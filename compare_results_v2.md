# So sánh Metric giữa 2 Notebook (Mới)

- **Notebook 1 (NB1):** [result_log.ipynb](file:///D:/Sau_Benh_object/OW_OVD-An-custom/New_retrival(01)/result_kaggle/result_log.ipynb) tại New_retrival(01)/result_kaggle
- **Notebook 2 (NB2):** [ketqua4task_1epocj.ipynb](file:///D:/Sau_Benh_object/OW_OVD-An-custom/NewRetrieval_02/ketqua4task_1epocj.ipynb) tại NewRetrieval_02

## 1. So sánh Metric Huấn luyện & Validation (Training & Validation Metrics)
Đây là các metric sinh ra cuối epoch 1 trong quá trình train.

### Task 1
| Metric | NB1: result_log | NB2: ketqua4task_1epocj | Đánh giá / Chênh lệch (NB2 - NB1) |
| :--- | :---: | :---: | :--- |
| `coco/Prev class AP50` | - | - | - |
| `coco/Prev class Recall50` | - | - | - |
| `coco/Current class AP50` | - | 4.8687 | - |
| `coco/Current class Recall50` | - | 98.0066 | - |
| `coco/Known AP50` | - | 4.8687 | - |
| `coco/Known Recall50` | - | 98.0066 | - |
| `coco/Unknown AP50` | - | 0.0000 | - |
| `coco/Unknown Recall50` | - | 0.0000 | - |

### Task 2
| Metric | NB1: result_log | NB2: ketqua4task_1epocj | Đánh giá / Chênh lệch (NB2 - NB1) |
| :--- | :---: | :---: | :--- |
| `coco/Prev class AP50` | - | 8.2824 | - |
| `coco/Prev class Recall50` | - | 98.3558 | - |
| `coco/Current class AP50` | - | 3.1614 | - |
| `coco/Current class Recall50` | - | 96.3254 | - |
| `coco/Known AP50` | - | 5.9189 | - |
| `coco/Known Recall50` | - | 97.4187 | - |
| `coco/Unknown AP50` | - | 0.0000 | - |
| `coco/Unknown Recall50` | - | 0.0000 | - |

### Task 3
| Metric | NB1: result_log | NB2: ketqua4task_1epocj | Đánh giá / Chênh lệch (NB2 - NB1) |
| :--- | :---: | :---: | :--- |
| `coco/Prev class AP50` | - | 12.9922 | - |
| `coco/Prev class Recall50` | - | 97.4231 | - |
| `coco/Current class AP50` | - | 5.8327 | - |
| `coco/Current class Recall50` | - | 96.4660 | - |
| `coco/Known AP50` | - | 10.7313 | - |
| `coco/Known Recall50` | - | 97.1209 | - |
| `coco/Unknown AP50` | - | 0.0000 | - |
| `coco/Unknown Recall50` | - | 0.0000 | - |

### Task 4
| Metric | NB1: result_log | NB2: ketqua4task_1epocj | Đánh giá / Chênh lệch (NB2 - NB1) |
| :--- | :---: | :---: | :--- |
| `coco/Prev class AP50` | - | 27.1486 | - |
| `coco/Prev class Recall50` | - | 97.8957 | - |
| `coco/Current class AP50` | - | 36.7904 | - |
| `coco/Current class Recall50` | - | 100.0000 | - |
| `coco/Known AP50` | - | 29.4626 | - |
| `coco/Known Recall50` | - | 98.4008 | - |
| `coco/Unknown AP50` | - | 0.0000 | - |
| `coco/Unknown Recall50` | - | 0.0000 | - |

## 2. So sánh Metric Đánh giá Truy vấn Ảnh (Retrieval Evaluation Metrics)
Đây là các metric đánh giá cuối cùng của hệ thống retrieval.

### Task 1
| Metric | NB1: result_log | NB2: ketqua4task_1epocj | Đánh giá / Chênh lệch (NB2 - NB1) |
| :--- | :---: | :---: | :--- |
| `Global mAP` | - | 0.0584 | - |
| `Recall@1` | 0.5371 | 0.1712 | **-0.3659** (Giảm 🔴) |
| `Recall@5` | 0.7979 | 0.3843 | **-0.4136** (Giảm 🔴) |
| `Recall@10` | 0.8964 | 0.4948 | **-0.4016** (Giảm 🔴) |
| `AUROC / OOD AUROC` | 0.9426 | 0.4649 | **-0.4777** (Giảm 🔴) |
| `FPR@TPR95 / OOD FPR` | 0.0414 | 0.9489 | **+0.9075** (Tăng 🟢) |
| `Plasticity` | - | 0.0625 | - |
| `Forgetting (mAP)` | - | 0.0000 (0.00%) | - |
| `Overall Change` | - | 0.0625 | - |

### Task 2
| Metric | NB1: result_log | NB2: ketqua4task_1epocj | Đánh giá / Chênh lệch (NB2 - NB1) |
| :--- | :---: | :---: | :--- |
| `Global mAP` | - | 0.0595 | - |
| `Recall@1` | 0.5331 | 0.1766 | **-0.3565** (Giảm 🔴) |
| `Recall@5` | 0.7940 | 0.3791 | **-0.4149** (Giảm 🔴) |
| `Recall@10` | 0.8894 | 0.4948 | **-0.3946** (Giảm 🔴) |
| `AUROC / OOD AUROC` | 0.9503 | 0.4883 | **-0.4620** (Giảm 🔴) |
| `FPR@TPR95 / OOD FPR` | 0.0203 | 0.9516 | **+0.9313** (Tăng 🟢) |
| `Plasticity` | - | 0.0371 | - |
| `Forgetting (mAP)` | - | 0.0000 (0.00%) | - |
| `Overall Change` | - | 0.0371 | - |

### Task 3
| Metric | NB1: result_log | NB2: ketqua4task_1epocj | Đánh giá / Chênh lệch (NB2 - NB1) |
| :--- | :---: | :---: | :--- |
| `Global mAP` | - | 0.0622 | - |
| `Recall@1` | 0.5369 | 0.1917 | **-0.3452** (Giảm 🔴) |
| `Recall@5` | 0.7954 | 0.3808 | **-0.4146** (Giảm 🔴) |
| `Recall@10` | 0.8884 | 0.5005 | **-0.3879** (Giảm 🔴) |
| `AUROC / OOD AUROC` | 0.9359 | 0.4960 | **-0.4399** (Giảm 🔴) |
| `FPR@TPR95 / OOD FPR` | 0.0279 | 0.9617 | **+0.9338** (Tăng 🟢) |
| `Plasticity` | - | 0.0401 | - |
| `Forgetting (mAP)` | - | 0.0000 (0.00%) | - |
| `Overall Change` | - | 0.0401 | - |

### Task 4
| Metric | NB1: result_log | NB2: ketqua4task_1epocj | Đánh giá / Chênh lệch (NB2 - NB1) |
| :--- | :---: | :---: | :--- |
| `Global mAP` | - | 0.0638 | - |
| `Recall@1` | 0.5428 | 0.1921 | **-0.3507** (Giảm 🔴) |
| `Recall@5` | 0.8026 | 0.4025 | **-0.4001** (Giảm 🔴) |
| `Recall@10` | 0.8830 | 0.5203 | **-0.3627** (Giảm 🔴) |
| `AUROC / OOD AUROC` | - | 0.5000 | - |
| `FPR@TPR95 / OOD FPR` | - | 1.0000 | - |
| `Plasticity` | - | 0.2180 | - |
| `Forgetting (mAP)` | - | 0.0000 (0.00%) | - |
| `Overall Change` | - | 0.2180 | - |
