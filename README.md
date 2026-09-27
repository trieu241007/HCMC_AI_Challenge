# HCMC AI Challenge 2026 — Tìm kiếm video đa phương thức

Đây là project sinh viên được thực hiện cho cuộc thi HCMC AI Challenge 2026. Mục tiêu của project là thử nghiệm cách tìm kiếm những khoảnh khắc phù hợp trong video dựa trên câu mô tả hoặc câu hỏi bằng tiếng Việt.

Thay vì chỉ sử dụng hình ảnh, project kết hợp nhiều loại thông tin trong video như hình ảnh, chữ xuất hiện trên màn hình, lời nói và các đối tượng được phát hiện. Đây là một project phục vụ cho cuộc thi, vì vậy code và cấu trúc hiện tại chủ yếu tập trung vào thử nghiệm và xử lý dữ liệu.

## Mục tiêu của project

Khi người dùng nhập một câu mô tả, hệ thống sẽ tìm ra những frame hoặc đoạn video có nội dung gần với câu mô tả đó. Ví dụ, người dùng có thể tìm các cảnh có một người đang cầm một đồ vật, một dòng chữ xuất hiện trên màn hình hoặc một hành động xảy ra trong video.

## Các dạng bài toán trong cuộc thi

- **KIS:** tìm frame hoặc khoảnh khắc phù hợp nhất với câu mô tả.
- **QA:** sử dụng nội dung video để hỗ trợ trả lời câu hỏi.
- **TRAKE:** tìm các sự kiện hoặc hành động xảy ra theo một trình tự nhất định.

## Pipeline xử lý

```text
Video
  ↓
Trích xuất các keyframe đại diện
  ↓
Tạo thông tin hình ảnh, OCR, ASR và object detection
  ↓
Xây dựng dữ liệu tìm kiếm bằng FAISS / Elasticsearch
  ↓
Nhận câu truy vấn và tìm các kết quả phù hợp
  ↓
Hiển thị frame và kết quả tìm kiếm
```

### 1. Trích xuất keyframe

Video được chia thành các cảnh nhỏ và chọn ra những frame đại diện. Việc này giúp giảm số lượng frame cần xử lý nhưng vẫn giữ lại những khoảnh khắc quan trọng trong video.

### 2. Bổ sung thông tin cho frame

Các keyframe được xử lý để tạo thêm nhiều loại thông tin:

- **Visual embedding:** biểu diễn nội dung hình ảnh dưới dạng vector.
- **OCR:** đọc chữ xuất hiện trong frame.
- **ASR:** chuyển lời nói trong video thành văn bản.
- **Object detection:** nhận diện các đối tượng như người, xe hoặc đồ vật.

### 3. Tìm kiếm

Các vector hình ảnh được tìm kiếm bằng FAISS. Những thông tin dạng văn bản và metadata được xử lý thêm bằng Elasticsearch. Kết quả từ các nguồn khác nhau được kết hợp để chọn ra những frame phù hợp hơn với câu truy vấn.

### 4. Giao diện thử nghiệm

Project có một backend đơn giản bằng FastAPI và giao diện React để gửi câu truy vấn, xem kết quả và kiểm tra các frame lân cận.

## Các notebook xử lý trên Kaggle

Các notebook được sắp xếp theo thứ tự xử lý chính:

1. `01_extract_keyframes.ipynb` — trích xuất keyframe từ video.
2. `02_benchmark_keyframe_embeddings.ipynb` — thử nghiệm và tạo embedding cho keyframe.
3. `03_extract_ocr_text.ipynb` — đọc chữ trong hình ảnh.
4. `04_extract_asr_text.ipynb` — chuyển lời nói trong video thành văn bản.
5. `05_detect_objects.ipynb` — phát hiện các đối tượng trong keyframe.

Kaggle được sử dụng vì project cần GPU để xử lý số lượng video và keyframe lớn hơn trong thời gian hợp lý.

## Cấu trúc thư mục

- `kaggle/`: các notebook xử lý dữ liệu trên Kaggle.
- `search/`: code tìm kiếm và xếp hạng kết quả.
- `build/`: code xây dựng index và các thành phần hỗ trợ tìm kiếm.
- `build/ui/`: backend FastAPI và giao diện React thử nghiệm.
- `query/`: một số file truy vấn mẫu cho các bài toán của cuộc thi.

## Công nghệ sử dụng

Python, PyTorch, FAISS, Elasticsearch, FastAPI, React và Kaggle.

## Giới hạn của repository

Các video, keyframe, embedding, model file, search index và log kết quả không được đưa lên repository vì đây là dữ liệu của cuộc thi và có dung lượng lớn. Khi chạy lại project, cần chuẩn bị các dữ liệu và model tương ứng ở máy local hoặc trên Kaggle.

## Những điều học được

Project giúp thực hành quy trình xây dựng một hệ thống tìm kiếm video từ đầu đến cuối: xử lý dữ liệu, chạy các bước inference trên GPU, kết hợp nhiều loại thông tin và xây dựng một giao diện đơn giản để kiểm tra kết quả.
