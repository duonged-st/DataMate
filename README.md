# DataMate V5

V5 đọc CSV/Excel/SQLite của từng doanh nghiệp, chuẩn hóa cột rồi xây catalog SQLite gồm schema, nguồn file/sheet, mô tả README, khoảng ngày, trạng thái và khóa nối gợi ý. Qwen sinh SQL, Python chạy ở chế độ chỉ đọc và thử sửa tối đa một lần. Mỗi kết quả và log được ghi ngay. Đáp án Training chỉ được mở bởi lệnh `score`.

Inference dùng **Qwen3.5-9B** (9B tham số, Apache-2.0, trọng số công bố 01/03/2026 theo lịch sử kho model). [Model gốc](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) · [Q4_K_M GGUF, revision cố định](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf). Runtime là [llama.cpp v1.0.0](https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0). GGUF được kiểm tra SHA-256 trước khi dùng để build. Không fine-tune hay sinh dữ liệu tổng hợp.

## Kaggle

1. Bật GPU T4, thêm dataset code V5 và Training. Notebook nhận thư mục đã giải nén hoặc tên ZIP như `DataMate_V5 (1).zip`. Nếu có nhiều dataset trùng, điền `PROJECT` và `TRAIN` ở cell đầu. Thêm cache `llamacuda` cũ để dùng lại executable.
2. Mở `DataMate_V5_Kaggle.ipynb`, chạy cell chuẩn bị runtime, rồi chạy smoke 10 câu và regression đúng 80 ID V4.2. Mốc V4.2 là +13 điểm, 30 đúng, 17 sai và 33 abstain.
3. Điền các biz Training team chưa tối ưu vào `HOLDOUT_BIZ`, chạy benchmark và xuất `datamate_benchmarks.zip`. Mỗi lượt lưu cấu hình GPU, checksum model và code, danh sách câu hỏi, submission, điểm và log.
4. Giải nén ZIP kết quả rồi dùng `prepare_assets.py --benchmarks <thư mục đã giải nén>` khi chuẩn bị Docker để kèm log và cập nhật manifest.

### Quy tắc SQL bổ sung

Khi schema và câu hỏi khớp rõ ràng, pipeline dùng truy vấn theo rulebook cho doanh thu cùng kỳ tháng trước, tỷ lệ hủy đơn, khách mua nhiều lần nhất theo tháng và tổng tiền hoặc lượng bán của một mặt hàng được nêu tên. Nhánh này lấy tên bảng, cột, giá trị và đơn vị từ SQLite/catalog; nếu truy vấn không tạo được đáp án hợp lệ thì chuyển sang Qwen. Log ghi `source: rulebook_sql`, SQL và kết quả. Nhánh dùng quy tắc không sinh token nên không có logprob.

Nhánh Qwen nhận thêm tên sản phẩm tìm thấy, gợi ý hình dạng đáp án, mốc thời gian và các hàm trạng thái. Phép chia trong SQL percent được ép về số thực. Log ghi cả `generated_sql` lẫn SQL đã thực thi nếu có thay đổi. Smoke yêu cầu logprob và ghi `confidence_diagnostics`: `valid`, `request_rejected`, `missing_logprobs` hoặc `invalid_logprobs` theo lượt gọi. 

Trước khi thực thi, bộ kiểm tra phát hiện các mẫu lọc tháng sai và đưa lỗi vào lượt sửa SQL. Khi SQLite báo sai tên bảng hoặc cột, lượt sửa nhận lại danh sách định danh thực tế. Các truy vấn theo quy tắc nhận diện thêm số khách hoặc mặt hàng đạt ngưỡng số đơn, tỷ trọng doanh thu theo nhóm hàng, công nợ đến một ngày, giá giao dịch lịch sử và so sánh theo ca hoặc theo ngày đổi giá. Các truy vấn này chỉ chạy khi catalog có các bảng và cột tương ứng.

Nhánh `catalog_evidence` chỉ trả `unanswerable` cho trường hợp có bằng chứng về phạm vi ngày hoặc loại dữ liệu thiếu mà bộ kiểm tra nhận diện được. Nếu Qwen đề xuất `unanswerable` mà chưa xác minh được, pipeline đưa khoảng ngày và timestamp đã quan sát vào lượt sửa. Nếu vẫn thiếu căn cứ để trả lời, pipeline ghi `abstain` và lưu lý do trong log.

Nếu tên cột khác tên thường gặp, pipeline dùng mô tả cột trong README để tìm các vai trò như ngày bán, mã đơn, trạng thái và thành tiền dòng hàng. Với câu hỏi trung bình mỗi đơn, pipeline cộng thành tiền từng đơn rồi mới lấy trung bình. Khi chọn dòng hàng từ bảng riêng, pipeline kiểm tra khóa nối trong catalog và lọc theo tập mã đơn hợp lệ. Sổ bán không có mã đơn vẫn có thể tính tổng doanh thu theo dòng. Bộ kiểm tra ngày yêu cầu Qwen sửa khi dùng tên cột ngày như một chuỗi SQL.

Bản sao SQLite dùng để truy vấn chuẩn hóa ngày văn bản và loại các dòng hàng trùng hệt có cùng mã đơn và số dòng. Với Excel/CSV, bộ nạp làm tương tự khi nhận diện được hai khóa này. Catalog xác minh thêm khóa đơn khác tên giữa các bảng và đưa tên cột thật vào prompt. Quy tắc bán hàng xử lý trạng thái giao thất bại và đổi trả. Các truy vấn lọc theo hoạt chất so khớp toàn bộ giá trị trong danh mục, theo đúng rulebook về thuốc phối hợp.

### Output để phân tích

Mỗi benchmark Training tạo các file sau trong `/kaggle/working/datamate_benchmarks/<phase>/`.

| File | Nội dung |
|---|---|
| `submission.jsonl` | Một dòng mỗi câu theo schema BTC. |
| `predictions.csv` | Câu hỏi, biz, đáp án trước và sau quyết định, SQL, output thực thi, các lần thử, latency và lý do abstain. |
| `evaluation.csv` | Các cột dự đoán, gold, `is_correct`, `status`, `score` và nhóm lỗi. Chỉ xuất khi có gold. |
| `logs/questions.jsonl` | SQL từng lần thử, kết quả truy vấn, output model, reasoning do runtime trả, candidate, quyết định cuối và lý do abstain. `thinking` là null khi runtime không trả reasoning. |
| `by_business.csv` | Đúng, sai, abstain, PA và coverage từng biz. |
| `metrics.json` | `total`, `correct`, `wrong`, `abstain`, `coverage`, `pa`, `avg_latency_ms`, `confidence_method`, `optimal_threshold`. |
| `confidence_sweep.csv` | PA và coverage trên hai nhóm biz tại mỗi ngưỡng. Xuất sau cell calibration/validation. |

PA local = (đúng − sai) / tổng số câu. Scorer local theo rulebook phát hành, hỗ trợ cột bảng theo tên hoặc vị trí, hàng bảng không xét thứ tự, list có thứ tự và alias từng phần tử/hàng từ gold. Kết quả cục bộ chưa được đối chiếu với scorer nội bộ BTC. Sai giá trị chưa đủ bằng chứng để gán nguyên nhân sẽ ghi `undetermined`. Câu chưa tạo được đáp án hợp lệ có candidate null và vẫn lưu SQL hoặc lỗi nếu đã có.

`--log-confidence` lấy `exp(mean(logprob))` từ các token do llama-server trả, ghi số token và logprob trung bình trong log. Khi runtime không hỗ trợ hoặc không trả logprob hợp lệ, confidence là null. Đây là tín hiệu xếp hạng chưa hiệu chỉnh. Câu thiếu confidence giữ quyết định gốc.

Trong cell calibration/validation, chọn hai nhóm biz Training riêng biệt mà team chưa dùng để sửa pipeline. Code chọn ngưỡng tối đa PA trên calibration, ưu tiên coverage khi hòa. Sau đó áp dụng ngưỡng cố định trên validation. `threshold_validation/` có submission sau ngưỡng, log giữ đáp án trước ngưỡng, CSV quét và metrics. Code kiểm tra hai nhóm tách theo biz và chạy cùng cấu hình model. Dùng kết quả validation để đánh giá lựa chọn đã cố định. Điều chỉnh lại ngưỡng theo validation sẽ dùng tập đó như calibration.

### Áp dụng ngưỡng trong bản nộp

Sau lượt calibration/validation, tải `datamate_threshold_validation.zip` và lấy `confidence_policy.json`. Báo cáo ghi AUC phân biệt đúng/sai và tỷ lệ đúng theo khoảng confidence. V5 chỉ xuất ngưỡng triển khai khi calibration có cả câu đúng lẫn câu sai có logprob, AUC lớn hơn 0,5, PA calibration tăng và ngưỡng cố định cũng tăng PA trên validation. AUC mô tả khả năng xếp hạng, còn PA đo tác động của việc bỏ câu. Nếu thiếu bằng chứng hoặc PA validation không tăng, config ghi `threshold: null`.

Khi chuẩn bị Docker, truyền thêm `--confidence-policy /path/to/confidence_policy.json` vào lệnh `prepare_assets.py`. Docker đóng gói config và `predict.sh` tự đọc khi BTC chạy. Nếu có ngưỡng, CLI tự yêu cầu logprob và áp dụng cùng một hàm lọc như bước đánh giá. Câu thiếu logprob giữ kết quả cũ. Log giữ `candidate_answer`, confidence và lý do abstain. Kiểm tra tỷ lệ có logprob và PA trong chính image sẽ nộp, đặc biệt khi executable Kaggle khác bản llama-server đóng gói. Bản ZIP mặc định có `threshold: null` vì chưa có policy đo từ Qwen thực.

Nếu còn file từ lượt cũ, có thể tạo báo cáo bằng CPU:

```bash
python report.py --data "/path/to/train" --output "/path/to/v5_dev80" --allow-missing-logs
```

Không có log thì các trường SQL, latency và confidence để trống. Dữ liệu không có gold chỉ tạo predictions và metrics mô tả lượt chạy khi gọi `report.py`.

## Máy Windows hoặc Linux chạy audit bằng CPU

Chạy từ thư mục đã giải nén:

```bash
python -m pip install -r requirements.txt
python -m datamate audit --data "public_test(1).zip" --output result
```

`loader_audit.json` ghi mapping file/sheet, dòng header, số dòng nhập và số ô không parse được. Thay đường dẫn ZIP bằng đường dẫn thật. Có thể truyền thư mục giải nén chứa `questions.jsonl`.

## Docker trên máy Linux có NVIDIA GPU

Chạy từ thư mục dự án. Chuẩn bị weights và bản sao Training trước khi build. Có thể thêm `--model /path/to/model.gguf` để dùng GGUF đã có.

```bash
python3 -m pip install -r requirements.txt
python3 prepare_assets.py --train "/path/to/train (1).zip"
TEAM=your_team_slug
IMAGE="zaic2026_datamate_${TEAM}:final"
docker build --platform linux/amd64 -t "$IMAGE" .
mkdir -p result
docker run --rm --gpus all --network none --shm-size=16g --entrypoint bash \
  -v "/absolute/path/to/public_test:/data:ro" -v "$PWD/result:/result" \
  "$IMAGE" /code/predict.sh
```

`/absolute/path/to/public_test` phải là thư mục chứa `questions.jsonl`, `rulebook.md`, `calendar.csv` và thư mục `data`. Dùng `--gpus '"device=0,1"'` trên máy có hai GPU để chạy đúng cách gắn GPU của BTC. Image sử dụng GPU đầu tiên. Để smoke test offline trước khi chạy cả split, lưu 10 dòng đầu của questions.jsonl vào một file tạm, rồi thêm bind mount read-only của file đó tại `/data/questions.jsonl` trong lệnh Docker. Ở runtime, llama-server dùng GGUF đã đóng gói và nhận truy vấn qua loopback. Output là `result/submission.jsonl` và `result/logs/questions.jsonl`.

Kiểm tra đủ ID và log. Chạy lại với thư mục kết quả khác rồi thêm `--compare result_run2` để so đáp án:

```bash
python3 verify_outputs.py --data "/absolute/path/to/public_test" --output result
```

Sau khi đạt yêu cầu, lưu image cuối:

```bash
set -o pipefail
docker save "$IMAGE" | gzip > "zaic2026_datamate_${TEAM}.tar.gz"
md5sum "zaic2026_datamate_${TEAM}.tar.gz" > "zaic2026_datamate_${TEAM}.tar.gz.md5"
gzip -dc "zaic2026_datamate_${TEAM}.tar.gz" | wc -c
```

## Phạm vi đã xác minh

Xem `verification/public_summary.json`, `verification/train_summary.json` và `verification/cpu_smoke.json`. CPU smoke chạy SQL kiểm soát trên dữ liệu thật và đi qua prompt, repair, JSONL và log. Cần chạy Qwen thực trên Kaggle, rồi build và test Docker offline trên máy GPU trước khi chốt bài nộp. Pipeline quản lý tiến trình local. Mỗi câu có một lượt phục hồi kết nối, gồm khởi động lại tiến trình và thử lại truy vấn. Nếu vẫn lỗi, pipeline ghi `abstain` và chuyển sang câu sau. Bản Docker dành 5 phút cuối của giới hạn 6 giờ để ghi kết quả còn lại. Cột số lượng mơ hồ mà không suy ra được dấu phân cách được ghi NULL và báo trong audit. Các quy tắc ngữ nghĩa như công nợ, dòng voucher, khuyến mãi, hoàn đơn và cách nối bảng dựa vào schema và README. Điểm benchmark sẽ cho biết cần sửa nhóm nào tiếp.
