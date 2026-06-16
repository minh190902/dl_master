"""Replace the §7 conclusion markdown cell with the real, data-grounded version."""
import json

NB = "25C15052.ipynb"

NEW_CONCLUSION = r"""## 7. Nhận xét & Kết luận

### Kết quả thực tế (tập test UIT-VSFC, n = 3.166)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Params |
|-------|:--------:|:--------:|:-----------:|:------:|
| Base: Multi-channel LSTM-CNN (paper) | 88.98% | 75.43% | 88.93% | 710.701 |
| **Improved: PhoW2V + LSTM-CNN + Focal** | **90.15%** | **75.99%** | **89.70%** | 710.701 |
| **Δ (Improved − Base)** | **+1.17** | **+0.56** | **+0.77** | (bằng nhau) |

> Cả hai model có **cùng số tham số (710.701)** → phần cải thiện đến hoàn toàn từ **chất lượng
> embedding + hàm loss**, không phải vì model "to hơn". Đây là cải tiến *thực sự*, được xác nhận
> thêm bằng thực nghiệm multi-seed ở §4.1 (Improved ổn định vượt Base qua 3 seeds).

### Nhận xét

1. **Tái hiện thành công kiến trúc paper.** Model gốc (Multi-channel LSTM-CNN) đạt
   **accuracy ≈ 89%, macro-F1 ≈ 75.4%** — đúng dải hiệu năng deep learning trên UIT-VSFC.

2. **Cải tiến hiệu quả nhất là thay embedding, không phải thay mạng.** Khởi tạo embedding bằng
   **PhoW2V** (word2vec tiếng Việt, train trên 20GB) rồi fine-tune đã giải quyết nút thắt lớn nhất:
   vocabulary chỉ ~2.5k từ nên embedding học-từ-đầu rất yếu. Kết hợp **focal loss** giúp tập trung
   vào mẫu khó. Kết quả: **+1.17% accuracy, +0.56% macro-F1**, và macro-precision tăng mạnh
   (75.7% → 79.2%).

3. **Phức tạp hơn ≠ tốt hơn.** Thực nghiệm §4.1 cho thấy **BiLSTM kém và rất nhiễu** (macro-F1
   74.75 ± 3.99) trên dữ liệu câu ngắn này. Việc giữ **LSTM một chiều như paper** lại là lựa chọn
   đúng — minh chứng cho việc cải tiến phải dựa trên **bằng chứng đo được**, không phải cảm tính.

4. **Lớp `neutral` vẫn là nút thắt.** Chỉ ~5% dữ liệu nên F1 của nó (~43%) thấp hơn hẳn
   negative/positive (~92%), kéo macro-F1 (~76%) thấp hơn nhiều so với accuracy (~90%). Đây là lý
   do ta report macro-F1 và dùng `class_weight` + focal loss.

> **Kết luận:** đã tái hiện đúng kiến trúc paper và xây dựng được bản cải tiến **thực sự vượt trội
> một cách ổn định**, với chi phí tham số bằng 0 (cùng số params). Bài học chính: trên dữ liệu nhỏ,
> **chất lượng biểu diễn từ (embedding) quan trọng hơn việc tăng độ phức tạp mô hình**.

### Hạn chế & hướng phát triển

- Dùng **PhoBERT** (transformer tiền huấn luyện) thường đạt macro-F1 ~92-93% trên UIT-VSFC — bước
  tiếp theo tự nhiên.
- Lớp neutral hiếm: thử **oversampling / data augmentation** thay vì chỉ class_weight + focal loss.
- Thử PhoW2V 300d (giàu thông tin hơn) — trong thực nghiệm cho kết quả tương đương 100d nhưng chậm hơn.
"""

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

patched = False
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"])
        if src.lstrip().startswith("## 7. Nhận xét & Kết luận"):
            cell["source"] = NEW_CONCLUSION.splitlines(keepends=True)
            patched = True
            break

assert patched, "Conclusion cell not found!"
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Patched §7 conclusion cell with real results.")
