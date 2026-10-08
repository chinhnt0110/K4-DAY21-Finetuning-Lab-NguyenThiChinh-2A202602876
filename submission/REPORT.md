# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Thị Chinh  **MSSV**: 2A202602876  **Ngày**: 08/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 max_steps |

**Template có giữ khối `<think>` không?** `có` — *(results/template_check.json)*
Nếu không: bạn đã xử lý thế nào? `Không cần xử lý: chat template mặc định giữ nguyên cặp thẻ <think>...</think> và nội dung bên trong (verdict: reasoning preserved — safe to train on traces).`

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0 | 0.7911 | 0.0 | 3391.0 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.0 | 1028.5 |
| (c) LoRA fine-tune | 0.97 | 0.6778 | 1.0 | 1373.4 |

**(b) có thật sự mạnh hơn (a) không?** `có` — target tăng từ 0.0 lên 0.765, format JSON hợp lệ từ 0.0 lên 1.0, latency giảm từ 3391 ms xuống 1028.5 ms nhờ prompt định hướng output rõ ràng.
Bạn có sửa `OPTIMIZED_PROMPT` không? `Không. Giữ nguyên prompt gốc (SHA: 719e74d3b6232053) để baseline đối chứng công bằng.`

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6271 | 0.9700 | 417.2 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 | 0.5377 | 0.9700 | 285.8 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | 0.0000 | 410.8 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.9400 | 474.1 | 3.86 |

**4.1 — `attn_only` vs `correct`: thắng, thua hay hoà? Thứ tự có giống train loss không? Nói gì về *rank* so với *vị trí adapter*?**
Trên target, `attn_only` HOÀ với `correct` (cùng 0.9700). Nhưng train loss của `attn_only` thấp hơn (0.5377 so với 0.6271), nên thứ tự theo loss không phản ánh năng lực thực. Dồn rank rất cao (r=283) vào vài khối attention chỉ giúp ghi nhớ tập train tốt hơn, không tăng chất lượng tác vụ. Gắn adapter rộng trên toàn text-linear với r=16 cho kết quả tương đương mà gọn hơn.

**4.2 — `wrong_lr` chỉ khác một con số. Đường loss ra sao? Chỉ nhìn loss thì kết luận sai điều gì?**
`wrong_lr` dùng LR thang full fine-tune (1e-5 thay vì 1e-4), nên adapter cập nhật quá chậm. Loss dừng ở mức rất cao (1.5702 so với 0.6271 của `correct`), model không học được format JSON và đạt target 0.0000. Nếu chỉ nhìn loss mà không kiểm tra LR, ta dễ kết luận nhầm rằng LoRA thất bại hoặc dữ liệu không học được.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá gì? Có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` giảm hơn 56% VRAM (3.86 GB so với 8.78 GB). Cái giá: train lâu hơn gần 14% (474.1s so với 417.2s), target giảm từ 0.9700 xuống 0.9400, latency suy luận tăng lên 1782 ms. Số đo ủng hộ khuyến nghị của vendor: nếu VRAM 16-bit còn đủ thì không nên dùng QLoRA 4-bit cho Qwen3.5.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.113` · `valid_trace_rate = 0.0`

Diễn giải: Cổng FAILED là đúng. Target tăng mạnh (+0.205, từ 0.765 lên 0.970), nhưng năng lực tổng quát giảm -0.113, vượt xa ngưỡng dung sai 0.020. Nguyên nhân là SFT chỉ dùng ticket CSKH, không có dữ liệu tổng quát để giữ tri thức nền, dẫn đến quên thảm hoạ (catastrophic forgetting). Tối ưu một metric hẹp mà bỏ qua model nền sẽ phá hỏng tính tổng quát. Hướng sửa theo deck §6.3 là trộn 1–5% dữ liệu tổng quát (replay buffer) vào tập train, không phải nới ngưỡng của cổng.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | doi_tra, cao, chuột không dây, tich_cuc | doi_tra, trung_binh, chuột không dây, trung_tinh | doi_tra, cao, chuột không dây, tich_cuc | ✅ FT thắng: đúng cả 4 trường, nhất là urgency và sentiment |
| 2 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi... | hoan_tien, cao, đèn bàn LED, tich_cuc | hoan_tien, trung_binh, đèn bàn LED, trung_tinh | hoan_tien, cao, đèn bàn LED, tich_cuc | ✅ FT thắng: nhận ra "quá hạn rồi" là urgency cao |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện... | hoan_tien, thap, bình giữ nhiệt, tich_cuc | hoan_tien, thap, bình giữ nhiệt, tich_cuc | hoan_tien, trung_binh, bình giữ nhiệt, tich_cuc | ❌ **FT thua**: urgency đoán trung_binh thay vì thap |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện... | san_pham_loi, thap, nồi chiên không dầu, trung_tinh | san_pham_loi, thap, nồi chiên không dầu, trung_tinh | san_pham_loi, trung_binh, nồi chiên không dầu, trung_tinh | ❌ **FT thua**: "khi nào tiện" bị nhầm thành trung_binh |
| 5 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop... | san_pham_loi, thap, áo khoác gió, tich_cuc | san_pham_loi, thap, áo khoác gió, tich_cuc | san_pham_loi, trung_binh, áo khoác gió, tich_cuc | ❌ **FT thua**: lặp lại lỗi urgency mức thấp |

Có mẫu chung nào ở các ca FT thua không?
Cả ba ca thua đều sai ở `urgency`: khi ticket có cụm thong thả như "khi nào tiện" (nhãn `thap`), bản fine-tune thiên về `trung_binh`. Prompt (b) có quy tắc tường minh nên xử lý đúng các ca biên này.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** **Chưa nên deploy** bản fine-tune này cho hệ thống dùng chung vì nó vi phạm cổng hồi quy (-0.113). Nếu triển khai như một microservice chỉ phân loại ticket CSKH thì bản này đáng dùng: target đạt 97.0% và không còn tốn token cho prompt dài. Đòn bẩy quyết định của lab là **mask** và **learning rate**, không phải rank. Mask sai (tính loss cả prompt) khiến model học vẹt việc sinh lại câu hỏi; LR sai (thang full fine-tune) khiến model gần như không học. Vị trí adapter trên toàn text-linear là nền ổn định, không cần đánh đổi bằng rank quá lớn.

**Ba điều tôi học được:**
1. **Verify mask trước khi train**: giải mã ngược token nhãn để chứng minh loss chỉ tính trên câu trả lời; tin vào mặc định là nguyên nhân hàng đầu khiến model sinh rác.
2. **Không chấm bằng train loss**: `attn_only` có train loss đẹp nhất nhưng target chỉ ngang `correct`; metric trên tác vụ thật mới đáng tin.
3. **Prompt tốt là baseline đáng gờm**: prompt tối ưu đã đạt target 0.765 với format 1.0 mà không làm hỏng model nền.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Trộn 3% dữ liệu hội thoại tổng quát (replay) vào tập train để đưa regression về trong ngưỡng, giữ target cao và đủ điều kiện deploy.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/chinhnt0110/Qwen3.5-4B-ticket-triage-lora

