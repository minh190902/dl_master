"""Replace the §7 conclusion markdown cell with the real, data-grounded version."""
import json

NB = "25C15052.ipynb"

NEW_CONCLUSION = r"""## 7. Nhận xét và Kết luận

### 7.1. Kết quả trên tập kiểm thử (UIT-VSFC, n = 3.166)

| Mô hình | Accuracy | Macro-F1 | Weighted-F1 | Số tham số |
|---------|:--------:|:--------:|:-----------:|:----------:|
| Mô hình gốc (Multi-channel LSTM-CNN) | 88.98% | 75.43% | 88.93% | 710.701 |
| **Mô hình cải tiến (PhoW2V + LSTM-CNN + Focal)** | **90.15%** | **75.99%** | **89.70%** | 710.701 |
| **Chênh lệch (Δ)** | **+1.17** | **+0.56** | **+0.77** | bằng nhau |

Hai mô hình có cùng số lượng tham số (710.701), do đó phần cải thiện hiệu năng đến hoàn toàn từ
chất lượng biểu diễn từ và hàm mất mát, không phải từ việc tăng dung lượng mô hình. Mức cải thiện
này còn được xác nhận là ổn định qua thực nghiệm trên nhiều hạt giống ngẫu nhiên ở §4.1.

### 7.2. Nhận xét

1. **Tái hiện thành công kiến trúc của bài báo.** Mô hình gốc đạt độ chính xác khoảng 89% và
   macro-F1 khoảng 75.4%, nằm trong khoảng hiệu năng thường thấy của các mô hình học sâu trên UIT-VSFC.

2. **Cải thiện hiệu quả nhất đến từ biểu diễn từ, không phải từ kiến trúc mạng.** Việc khởi tạo
   lớp embedding bằng PhoW2V và tinh chỉnh đã khắc phục hạn chế lớn nhất của mô hình gốc: từ điển
   nhỏ khiến embedding học từ đầu có chất lượng thấp. Kết hợp với Focal loss, mô hình cải tiến tăng
   1.17 điểm phần trăm độ chính xác và 0.56 điểm phần trăm macro-F1, trong đó macro-precision tăng
   đáng kể (từ 75.7% lên 79.2%).

3. **Tăng độ phức tạp không đồng nghĩa với hiệu năng cao hơn.** Thực nghiệm ở §4.1 cho thấy kiến
   trúc BiLSTM đạt kết quả thấp và kém ổn định (macro-F1 74.75 ± 3.99) trên loại dữ liệu câu ngắn.
   Việc giữ nguyên LSTM một chiều như bài báo là lựa chọn phù hợp, cho thấy quá trình cải tiến cần
   dựa trên bằng chứng định lượng.

4. **Lớp `neutral` là yếu tố hạn chế chính.** Do chỉ chiếm khoảng 5% dữ liệu, điểm F1 của lớp này
   (khoảng 43%) thấp hơn nhiều so với hai lớp còn lại (khoảng 92%), kéo macro-F1 (khoảng 76%) xuống
   thấp hơn đáng kể so với độ chính xác (khoảng 90%). Đây là lý do báo cáo sử dụng macro-F1 làm độ
   đo chính, đồng thời áp dụng trọng số lớp và Focal loss.

**Kết luận chung.** Báo cáo đã tái hiện chính xác kiến trúc Multi-channel LSTM-CNN của bài báo và
xây dựng một mô hình cải tiến đạt hiệu năng cao hơn một cách ổn định mà không làm tăng số lượng
tham số. Kết quả cho thấy đối với bộ dữ liệu có quy mô hạn chế, chất lượng biểu diễn từ đóng vai
trò quan trọng hơn việc gia tăng độ phức tạp của mô hình.

### 7.3. Hạn chế và hướng phát triển

- Áp dụng các mô hình ngôn ngữ tiền huấn luyện dạng Transformer (chẳng hạn PhoBERT), vốn thường đạt
  macro-F1 khoảng 92–93% trên UIT-VSFC.
- Xử lý lớp thiểu số `neutral` triệt để hơn bằng các kỹ thuật tăng cường dữ liệu (oversampling,
  data augmentation) thay vì chỉ dựa vào trọng số lớp và Focal loss.
- Khảo sát PhoW2V 300 chiều; trong các thực nghiệm hiện tại, phiên bản này cho kết quả tương đương
  phiên bản 100 chiều nhưng có chi phí tính toán cao hơn.
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
