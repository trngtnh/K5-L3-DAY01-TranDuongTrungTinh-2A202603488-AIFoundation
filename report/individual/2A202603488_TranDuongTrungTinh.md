# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603488.
- **Họ và tên:** Trần Dương Trung Tính.
- **SEED**: `1370`.

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Đã có đủ ba tệp xuất; cần đặt cùng thư mục báo cáo khi nộp. Kết quả tự kiểm tra trong JSON đạt **15/15**, có đủ **10 khóa dự đoán**, SEED = 1370 và thử thách số 3. Thí nghiệm model lớn 4b chưa hoàn tất do lỗi quota API nên chưa có số liệu để so sánh model lớn với model nhỏ.

## 2. Kết quả chính

Dữ liệu sau bỏ trùng có **5,169 tin**, gồm 653 spam và 4,516 ham; chia thành train 3,619 tin, validation 775 tin và test 775 tin. SUB gồm 40 tin lấy từ test, cân bằng 20 spam và 20 ham.

| Phương án | F1 test | F1 SUB 40 | Train (s) | Tin/s cả lô | p95 (ms/tin) | USD/1,000 tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.916 | 0.857 | 6.167 | 32,463.723 | 0.059 | 0.000 |
| 2. BoW + MLP | 0.928 | 0.857 | 6.149 | 34,092.435 | 0.466 | 0.000 |
| 3a. Encoder MiniLM + kNN | 0.887 | 0.900 | 3.362 | 101.649 | 16.996 | 0.000 |
| 3b. Encoder MiniLM + LogReg | 0.932 | 0.895 | 3.708 | 1,413.305 | 7.147 | 0.000 |
| 3c. Encoder MiniLM + MLP head | 0.942 | 0.919 | 4.385 | 1,652.638 | 10.111 | 0.000 |
| 4a. LLM nhỏ zero-shot | — | 0.950 | 0.000 | 0.145 | 7,467.667 | 0.025 |
| 4c. LLM nhỏ few-shot | — | 0.974 | 0.000 | 0.142 | 12,068.537 | 0.063 |

LLM chỉ đo trên SUB; dấu “—” nghĩa là không đo trên toàn bộ test. So sánh chất lượng với LLM cần dùng cột F1 SUB. SUB chỉ có 40 tin nên chênh lệch nhỏ chưa đủ để khẳng định phương án nào tốt hơn ổn định.

Trên test, thêm lớp ẩn cho BoW tăng F1 từ 0.916 lên 0.928; trên embedding, F1 tăng từ 0.932 lên 0.942. Few-shot tăng F1 SUB từ 0.950 lên 0.974, còn chi phí 40 tin tăng từ khoảng 0.00100 lên 0.00253 USD, khoảng 2.54 lần nếu dùng giá trị chưa làm tròn trong JSON. Cả hai phương án LLM có `invalid=0`.

Thời gian chuẩn bị của embedding gồm encode tập train và huấn luyện head nếu có. Thông lượng LLM có tính cả khoảng nghỉ 5 giây giữa các request, còn p50/p95 lấy từ thời gian gọi API được ghi nhận. Các số đo phụ thuộc môi trường chạy; 0 USD trong bảng mô hình cục bộ nghĩa là không tính phí API.

## 3. Dự đoán và thực tế

1. **Stress test — `hard_eval` (14:37:12):** tôi dự đoán accuracy BoW + LogReg 70%, BoW + MLP 74%, kNN 76%, Encoder + LogReg 80%, Encoder + MLP head 82%, LLM zero-shot 88% và LLM phòng thủ 92%. Thực tế lần lượt là **52%, 54%, 70%, 64%, 64%, 98%, 100%**. BoW + MLP thấp hơn dự đoán 20 điểm phần trăm. Tôi đã đánh giá quá cao khả năng tổng quát của mô hình dùng BoW trên những cách viết khác dữ liệu train.

2. **F1 của LogReg — `ml_f1` (13:43:49):** dự đoán 0.83, thực tế **0.916**, cao hơn 8.6 điểm phần trăm. Tín hiệu từ vựng và sự tương đồng giữa train/test giúp mô hình tuyến tính hoạt động tốt hơn dự kiến; kết quả stress test cho thấy ưu thế này không áp dụng cho mọi kiểu tin.

3. **Độ trễ — `tradeoff` (14:36:58):** tôi dự đoán chỉ hai mô hình BoW đạt SLA 50 ms. Thực tế **cả năm mô hình BoW và embedding đều đạt SLA** trong lần đo này. Danh sách Pareto dự đoán khớp kết quả, nhưng nhận định về tốc độ embedding chưa phù hợp với môi trường chạy hiện tại.

## 4. Phân tích 50 case khó

| Phương án | F1 SUB thông thường | F1 50 case khó | Accuracy 50 case khó |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0.857 | 0.333 | 52% |
| BoW + MLP | 0.857 | 0.378 | 54% |
| Encoder + kNN | 0.900 | 0.727 | 70% |
| Encoder + LogReg | 0.895 | 0.640 | 64% |
| Encoder + MLP head | 0.919 | 0.640 | 64% |
| LLM nhỏ zero-shot | 0.950 | 0.983 | 98% |
| LLM nhỏ + prompt phòng thủ | Chưa đo riêng | 1.000 | 100% |

**BoW + LogReg tụt nhiều nhất**, giảm 52.4 điểm phần trăm F1. Test thông thường cùng nguồn với train, có nhiều mẫu gần nhau nên có thể đánh giá quá lạc quan khả năng xử lý kiểu tin mới.

BoW + LogReg khó nhất ở spam giọng tự nhiên, chỉ đúng 10%; BoW + MLP khó nhất ở né từ khóa, đúng 0%. Cả ba mô hình embedding khó nhất ở spam giọng tự nhiên, lần lượt đúng 40%, 30% và 40%. LLM zero-shot khó nhất ở nhóm “tưởng spam nhưng không phải”, đúng 90%; bản phòng thủ đạt 100% mọi nhóm trong lần thử này.

Hai case cụ thể của nhóm né từ khóa:

- **Case #29:** “C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd…” có nhãn spam nhưng BoW + MLP đoán ham. Chữ bị thay bằng số tạo token khác vocab, làm mất các tín hiệu từ vựng quen thuộc.
- **Case #30:** “F R E E e n t r y into our weekly draw, t x t WIN…” có nhãn spam nhưng BoW + MLP đoán ham. Việc tách chữ bằng khoảng trắng khiến tokenizer không nhận ra các từ như “free”, “entry”, “txt”.

Cả hai case trên thuộc nhóm mà BoW + MLP đúng 0/6. MLP vẫn nhận đầu vào BoW nên không tự khôi phục được thông tin đã mất ở bước biểu diễn.

**4a zero-shot đúng 49/50**, còn **4d có prompt phòng thủ đúng 50/50**. Cả hai đều đúng 100% nhóm prompt injection; lợi ích quan sát được của prompt phòng thủ là nhóm “tưởng spam nhưng không phải” tăng từ 90% lên 100%, tương ứng sửa thêm một case. Không thấy nhóm nào giảm, nhưng chưa thể khẳng định hiệu quả trên mọi dữ liệu.

Có thể chuẩn hóa cách viết biến dạng để hỗ trợ BoW và bổ sung bộ kiểm tra URL/domain độc lập cho hệ thống. Cần tránh chuẩn hóa quá mạnh làm sai nội dung ham. Bộ 50 case hữu ích để phát hiện điểm yếu và kiểm tra hồi quy, nhưng chưa đại diện cho phân phối spam thực tế.

Đã thêm năm case bonus; chưa có output đánh giá lại trên 55 case nên chưa kết luận model sai case mới nào.

## 5. Thử thách cá nhân

**Đề số 3:** đổi lớp ẩn MLP thành 4 và 512 nơ-ron; báo số tham số, F1 test và thời gian train; giải thích khi nào chọn mạng nhỏ.

Hai mạng dùng cùng dữ liệu, đặt lại `torch.manual_seed(SEED)` trước khi khởi tạo, huấn luyện 30 epoch với Adam, learning rate 0.001, batch size 64. Với đầu vào 2000 chiều, số tham số là `2000h + h + h + 1 = 2002h + 1`.

| Nơ-ron ẩn | Số tham số | F1 test | Thời gian train (s) |
| --- | ---: | ---: | ---: |
| 4 | 8,009 | 0.934 | 2.442 |
| 512 | 1,025,025 | 0.922 | 18.872 |

Mạng 512 nơ-ron có số tham số gấp khoảng **128 lần**, train lâu hơn **7.73 lần**, nhưng F1 thấp hơn **1.2 điểm phần trăm**. Trong lần thử này, tôi chọn mạng 4 nơ-ron vì chất lượng tốt hơn và cần ít tài nguyên hơn.

Tăng kích thước mạng không bảo đảm cải thiện chất lượng trên dữ liệu nhỏ. Chưa đủ bằng chứng kết luận mạng lớn overfit chỉ từ F1 test; cần xem train/validation loss và thử thêm seed. Mạng nhỏ phù hợp khi dữ liệu ít, tài nguyên hạn chế hoặc chất lượng gần tương đương mạng lớn.

## 6. Phương án cho sản phẩm thật

Cả hai mô hình BoW và ba mô hình embedding đều đạt **p95 < 50 ms/tin**. Tôi ưu tiên **Encoder + MLP head** làm phương án ban đầu vì có F1 test cao nhất trong nhóm đạt SLA (**0.942**), p95 **10.111 ms**, thông lượng **1,652.638 tin/s**, thời gian chuẩn bị/train **4.385 giây**, không có phí API trong bảng.

Encoder + LogReg là phương án thay thế nhanh hơn, với F1 0.932 và p95 7.147 ms. kNN đạt SLA độ trễ nhưng thông lượng đo được 101.649 tin/s thấp hơn mức trung bình khoảng 116 tin/s. Cần kiểm thử tải đồng thời và giờ cao điểm trước khi triển khai; F1 stress test 0.640 của Encoder + MLP head cũng cho thấy cần cải thiện độ bền.

Với tháng 30 ngày, tổng lượng xử lý là **300 triệu SMS**. Dùng đơn giá chưa làm tròn trong JSON:

| Phương án LLM | USD/1,000 tin | Chi phí ước tính/tháng |
| --- | ---: | ---: |
| Nhỏ zero-shot | 0.0249033333 | 7,471 USD |
| Nhỏ few-shot | 0.0633033333 | 18,991 USD |

Công thức: `300,000,000 / 1,000 x USD/1,000 tin`. Đây là ước tính theo quy tắc token của lab, chưa tính đầy đủ token suy luận, retry và hạ tầng.

Tôi đề xuất router kết hợp **Encoder + MLP head và LLM**. Mô hình chính xử lý các tin đủ chắc chắn; tin có xác suất trong vùng lưỡng lự, ví dụ [0.3, 0.7], hoặc dấu hiệu bất thường được chuyển sang LLM để kiểm tra bổ sung. Ngưỡng cần chọn trên validation vì mô hình vẫn có thể đoán sai rất tự tin.

Nếu chỉ 1% lưu lượng chuyển sang LLM nhỏ zero-shot, phí API quy đổi còn khoảng **74.71 USD/tháng**. Tuy nhiên, hơn 5% request phải chờ LLM trên 50 ms sẽ khiến p95 chung không đạt SLA; cần kiểm soát tỷ lệ này hoặc xử lý LLM bất đồng bộ nếu nghiệp vụ cho phép.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 0–3 | Hỏi khái niệm, cách triển khai và giải thích kết quả | Giải thích lý thuyết, gợi ý dự đoán, bản nháp nhận xét | Chạy notebook, đối chiếu các ô kiểm tra, loss và số liệu thực nghiệm |
| Phần 4 — LLM | Hỏi cách dùng Gemini và xử lý lỗi API | Hướng dẫn cấu hình, API, prompt, parser và xử lý giới hạn request | Chạy thử API, so sánh zero-shot/few-shot và kiểm tra đầu ra |
| Phần 5 — Tổng hợp | Hỏi cách phân tích kết quả, tạo case và viết báo cáo | Hỗ trợ phân tích số liệu, nhận xét và báo cáo Markdown | Đối chiếu notebook và JSON, đồng bộ số liệu; chưa có kết quả chạy các case mới |
| Tính tay | Hỏi công thức và cách trình bày | Hướng dẫn các bước tính | Đối chiếu năm giá trị HAND_CALC, có ảnh viết tay kèm các bước |
| Thử thách cá nhân | Hỏi cách so sánh mạng 4 và 512 nơ-ron | Hỗ trợ giải thích kết quả | Chạy thử, đối chiếu số tham số, F1 và thời gian huấn luyện |

Tôi hiểu rằng điểm test cao trên dữ liệu cùng nguồn chưa bảo đảm xử lý tốt các kiểu spam mới. Tôi cũng nhận ra mạng lớn hơn không đồng nghĩa với việc luôn tốt hơn và cần chọn mô hình bằng nhiều tiêu chí như chất lượng, độ trễ, thông lượng, chi phí.... Khi đọc kết quả, phải phân biệt tập test, SUB và stress test, đồng thời cập nhật nhận xét theo đúng output của lần chạy đang báo cáo.
