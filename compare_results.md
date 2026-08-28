# So sánh Metric giữa 2 Notebook

- **Notebook 1 (NB1):** [log huấn luyện 1 epoch.ipynb](file:///D:/Sau_Benh_object/OW_OVD-An-custom/New_retrival(01)/log huấn luyện 1 epoch.ipynb)
- **Notebook 2 (NB2):** [ketqua4task_1epocj.ipynb](file:///D:/Sau_Benh_object/OW_OVD-An-custom/NewRetrieval_02/ketqua4task_1epocj.ipynb)

> [!NOTE]
> Notebook 1 bị lỗi crash ở cuối phần evaluation cho các task (`FileNotFoundError` cho `fNew_retrival/query_cache_task4_new_model.pkl`), nên không có kết quả đánh giá (Retrieval Evaluation Metrics). Dưới đây vẫn liệt kê đầy đủ để so sánh.

## 1. So sánh Metric Huấn luyện & Validation (Training & Validation Metrics)
Đây là các metric sinh ra cuối epoch 1 trong quá trình train (dùng bộ validation của YOLO-World/MMengine).

### Task 1
| Metric | NB1: log huấn luyện 1 epoch | NB2: ketqua4task_1epocj | Đánh giá / Chênh lệch (NB2 - NB1) |
| :--- | :---: | :---: | :--- |
| `coco/Prev class AP50` | - | - | - |
| `coco/Prev class Recall50` | - | - | - |
| `coco/Current class AP50` | 3.7236 | 4.8687 | **+1.1451** (Tăng 🟢) |
| `coco/Current class Recall50` | 83.6981 | 98.0066 | **+14.3085** (Tăng 🟢) |
| `coco/Known AP50` | 3.7236 | 4.8687 | **+1.1451** (Tăng 🟢) |
| `coco/Known Recall50` | 83.6981 | 98.0066 | **+14.3085** (Tăng 🟢) |
| `coco/Unknown AP50` | 0.0000 | 0.0000 | 0.0000 (Không đổi) |
| `coco/Unknown Recall50` | 0.0000 | 0.0000 | 0.0000 (Không đổi) |

### Task 2
| Metric | NB1: log huấn luyện 1 epoch | NB2: ketqua4task_1epocj | Đánh giá / Chênh lệch (NB2 - NB1) |
| :--- | :---: | :---: | :--- |
| `coco/Prev class AP50` | 1.6540 | 8.2824 | **+6.6284** (Tăng 🟢) |
| `coco/Prev class Recall50` | 45.5091 | 98.3558 | **+52.8467** (Tăng 🟢) |
| `coco/Current class AP50` | 0.8468 | 3.1614 | **+2.3146** (Tăng 🟢) |
| `coco/Current class Recall50` | 13.3283 | 96.3254 | **+82.9971** (Tăng 🟢) |
| `coco/Known AP50` | 1.2292 | 5.9189 | **+4.6897** (Tăng 🟢) |
| `coco/Known Recall50` | 28.5719 | 97.4187 | **+68.8468** (Tăng 🟢) |
| `coco/Unknown AP50` | 0.0000 | 0.0000 | 0.0000 (Không đổi) |
| `coco/Unknown Recall50` | 0.0000 | 0.0000 | 0.0000 (Không đổi) |

### Task 3
| Metric | NB1: log huấn luyện 1 epoch | NB2: ketqua4task_1epocj | Đánh giá / Chênh lệch (NB2 - NB1) |
| :--- | :---: | :---: | :--- |
| `coco/Prev class AP50` | 0.3189 | 12.9922 | **+12.6733** (Tăng 🟢) |
| `coco/Prev class Recall50` | 15.6555 | 97.4231 | **+81.7676** (Tăng 🟢) |
| `coco/Current class AP50` | 0.0000 | 5.8327 | **+5.8327** (Tăng 🟢) |
| `coco/Current class Recall50` | 0.0000 | 96.4660 | **+96.4660** (Tăng 🟢) |
| `coco/Known AP50` | 0.2150 | 10.7313 | **+10.5163** (Tăng 🟢) |
| `coco/Known Recall50` | 10.5584 | 97.1209 | **+86.5625** (Tăng 🟢) |
| `coco/Unknown AP50` | 0.0000 | 0.0000 | 0.0000 (Không đổi) |
| `coco/Unknown Recall50` | 0.0000 | 0.0000 | 0.0000 (Không đổi) |

### Task 4
| Metric | NB1: log huấn luyện 1 epoch | NB2: ketqua4task_1epocj | Đánh giá / Chênh lệch (NB2 - NB1) |
| :--- | :---: | :---: | :--- |
| `coco/Prev class AP50` | 0.3192 | 27.1486 | **+26.8294** (Tăng 🟢) |
| `coco/Prev class Recall50` | 8.4089 | 97.8957 | **+89.4868** (Tăng 🟢) |
| `coco/Current class AP50` | 0.0076 | 36.7904 | **+36.7828** (Tăng 🟢) |
| `coco/Current class Recall50` | 0.2466 | 100.0000 | **+99.7534** (Tăng 🟢) |
| `coco/Known AP50` | 0.2486 | 29.4626 | **+29.2140** (Tăng 🟢) |
| `coco/Known Recall50` | 6.5588 | 98.4008 | **+91.8420** (Tăng 🟢) |
| `coco/Unknown AP50` | 0.0000 | 0.0000 | 0.0000 (Không đổi) |
| `coco/Unknown Recall50` | 0.0000 | 0.0000 | 0.0000 (Không đổi) |

## 2. So sánh Metric Đánh giá Truy vấn Ảnh (Retrieval Evaluation Metrics)
Đây là các metric đánh giá cuối cùng sau khi hoàn thành huấn luyện.

### Task 1
| Metric | NB1: log huấn luyện 1 epoch | NB2: ketqua4task_1epocj |
| :--- | :---: | :---: |
| `Global mAP` | - | 0.0584 |
| `Recall@1` | - | 0.1712 |
| `Recall@5` | - | 0.3843 |
| `Recall@10` | - | 0.4948 |
| `OOD AUROC` | - | 0.4649 |
| `OOD FPR@TPR95` | - | 0.9489 |
| `Plasticity` | - | 0.0625 |
| `Forgetting (mAP)` | - | 0.0000 (0.00%) |
| `Overall Change` | - | 0.0625 |

### Task 2
| Metric | NB1: log huấn luyện 1 epoch | NB2: ketqua4task_1epocj |
| :--- | :---: | :---: |
| `Global mAP` | - | 0.0595 |
| `Recall@1` | - | 0.1766 |
| `Recall@5` | - | 0.3791 |
| `Recall@10` | - | 0.4948 |
| `OOD AUROC` | - | 0.4883 |
| `OOD FPR@TPR95` | - | 0.9516 |
| `Plasticity` | - | 0.0371 |
| `Forgetting (mAP)` | - | 0.0000 (0.00%) |
| `Overall Change` | - | 0.0371 |

### Task 3
| Metric | NB1: log huấn luyện 1 epoch | NB2: ketqua4task_1epocj |
| :--- | :---: | :---: |
| `Global mAP` | - | 0.0622 |
| `Recall@1` | - | 0.1917 |
| `Recall@5` | - | 0.3808 |
| `Recall@10` | - | 0.5005 |
| `OOD AUROC` | - | 0.4960 |
| `OOD FPR@TPR95` | - | 0.9617 |
| `Plasticity` | - | 0.0401 |
| `Forgetting (mAP)` | - | 0.0000 (0.00%) |
| `Overall Change` | - | 0.0401 |

### Task 4
| Metric | NB1: log huấn luyện 1 epoch | NB2: ketqua4task_1epocj |
| :--- | :---: | :---: |
| `Global mAP` | - | 0.0638 |
| `Recall@1` | - | 0.1921 |
| `Recall@5` | - | 0.4025 |
| `Recall@10` | - | 0.5203 |
| `OOD AUROC` | - | 0.5000 |
| `OOD FPR@TPR95` | - | 1.0000 |
| `Plasticity` | - | 0.2180 |
| `Forgetting (mAP)` | - | 0.0000 (0.00%) |
| `Overall Change` | - | 0.2180 |
