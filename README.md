# Hệ Thống Phân Loại Viêm Phổi Từ Ảnh X-Quang Lồng Ngực (PyTorch & CustomTkinter)

Dự án Học Sâu (Deep Learning) toàn diện về bài toán Thị giác máy tính trong Y tế (Medical Computer Vision): **Phát hiện Viêm Phổi (PNEUMONIA) và Phổi Bình Thường (NORMAL) từ ảnh chụp X-quang lồng ngực** sử dụng **PyTorch thuần**.

Dự án xây dựng một quy trình hoàn chỉnh từ dữ liệu ảnh y khoa thô, kỹ thuật tăng cường dữ liệu (Data Augmentation), xử lý mất cân bằng lớp dữ liệu (Class Imbalance) bằng hàm mất mát `BCEWithLogitsLoss` kết hợp `pos_weight`, huấn luyện đánh giá mô hình CNN theo kiến trúc Sigmoid (1 nơ-ron đầu ra), cho đến việc đóng gói thành **Ứng dụng Desktop (GUI App) hoàn chỉnh** bằng thư viện **CustomTkinter** với chế độ nền tối (Dark Mode) hiện đại, hỗ trợ tự động chẩn đoán và tự động huấn luyện lại (Re-Train) ngầm ngay trên giao diện.

---

## 1. Tổng Quan Bài Toán

Viêm phổi (Pneumonia) là một trong những căn bệnh nhiễm trùng đường hô hấp cấp tính nguy hiểm hàng đầu thế giới, đặc biệt ở trẻ nhỏ và người cao tuổi. Việc chẩn đoán sớm thông qua phim chụp X-quang lồng ngực (Chest X-Ray) đóng vai trò sống còn trong phác đồ điều trị của các bác sĩ.

Dự án này giải quyết bài toán phân loại nhị phân (Binary Image Classification):
- **Nhãn 0 (NORMAL - Bình Thường):** Phổi khỏe mạnh, các phế nang thông thoáng, không có tổn thương đông đặc hay thâm nhiễm.
- **Nhãn 1 (PNEUMONIA - Viêm Phổi):** Phổi có dấu hiệu bệnh lý, xuất hiện các đốm trắng mờ hoặc vùng đông đặc do vi khuẩn hoặc virus gây ra.

**Mục tiêu cốt lõi:**
1. Xây dựng mạng nơ-ron tích chập (CNN) nhị phân đầu ra 1 nơ-ron (Sigmoid), tối ưu hóa tốc độ và khả năng tính toán xác suất bệnh lý trực tiếp (0% - 100%).
2. Giải quyết triệt để vấn đề mất cân bằng dữ liệu y khoa (số ca viêm phổi nhiều gấp ~3 lần ca bình thường) bằng hệ số cân bằng `pos_weight`.
3. Tối đa hóa chỉ số **Recall cho ca Viêm Phổi (đạt 98.72%)** nhằm giảm thiểu tối đa tình trạng bỏ sót bệnh nhân nguy kịch (False Negative).
4. Đóng gói sản phẩm thành công cụ Desktop trực quan, cho phép người dùng nạp thêm ảnh X-quang mới vào thư mục và nhấn nút huấn luyện lại mô hình tức thì.

---

## 2. Cấu Trúc Thư Mục Dự Án

```text
Chest_Xray_Pneumonia/
├── assets/                  # Logo và biểu tượng phổi y tế trong suốt (PNG, ICO)
│   ├── lungs_icon.png
│   └── lungs_icon.ico
├── chest_xray/              # Toàn bộ bộ dữ liệu ảnh X-quang (>5.800 ảnh)
│   └── chest_xray/
│       ├── train/           # NORMAL (1.341 ảnh) | PNEUMONIA (3.875 ảnh)
│       ├── val/             # Tập kiểm định trong quá trình huấn luyện
│       └── test/            # Tập kiểm thử độc lập (624 ảnh)
├── notebooks/               # Jupyter Notebooks nghiên cứu & thực nghiệm mô hình
│   ├── NTT_ChestXRay_Sigmoid.ipynb
│   └── NTT_ChestXRay_Pneumonia.ipynb
├── saved_models/            # Nơi lưu trữ checkpoint mô hình đã huấn luyện
│   └── best_chest_xray_model.pth
├── src/                     # Mã nguồn huấn luyện độc lập qua dòng lệnh
│   └── train.py             # Script huấn luyện CLI & lưu model
├── du_doan_viem_phoi.py     # Ứng dụng Desktop GUI hoàn chỉnh (CustomTkinter)
├── requirements.txt         # Danh sách thư viện cần thiết
├── .gitignore               # Cấu hình bỏ qua cache và dataset lớn khi push Git
└── README.md                # Tài liệu hướng dẫn sử dụng dự án
```

---

## 3. Quy Trình Kỹ Thuật (Pipeline)

- **Bước 1 (Chuẩn bị & Tăng cường dữ liệu - Data Augmentation):**
  - Đọc dữ liệu theo cấu trúc chuẩn `torchvision.datasets.ImageFolder`.
  - Resize ảnh về kích thước đồng nhất `(128, 128)`.
  - Áp dụng các phép biến đổi tăng cường ngẫu nhiên: Xoay góc nhẹ (`RandomRotation(10)`), Lật ngang (`RandomHorizontalFlip()`) nhằm tăng khả năng tổng quát hóa, chống phụ thuộc vào tư thế chụp X-quang của bệnh nhân.
  - Chuẩn hóa phân phối điểm ảnh theo chuẩn ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Bước 2 (Kiến trúc mạng tích chập - ChestXRayBinaryNet):**
  - **Khối Conv 1:** `Conv2D(in=3, out=32, k=3, p=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`. Kích thước đặc trưng: `32 x 64 x 64`.
  - **Khối Conv 2:** `Conv2D(in=32, out=64, k=3, p=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`. Kích thước đặc trưng: `64 x 32 x 32`.
  - **Khối Conv 3:** `Conv2D(in=64, out=128, k=3, p=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)`. Kích thước đặc trưng: `128 x 16 x 16`.
  - **Khối Phân loại (Fully Connected):** Duỗi phẳng vector $128 \times 16 \times 16 = 32.768$ chiều $\rightarrow$ `Linear(32768, 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(128, 1)`.
- **Bước 3 (Xử lý mất cân bằng lớp - Class Imbalance):**
  - Trong tập dữ liệu huấn luyện, số lượng ảnh `NORMAL` là 1.341 và `PNEUMONIA` là 3.875 (tỷ lệ xấp xỉ $1 : 2.89$).
  - Tính hệ số cân bằng trọng số cho lớp dương (Viêm phổi):
    $$\text{pos\_weight} = \frac{N_{\text{NORMAL}}}{N_{\text{PNEUMONIA}}} = \frac{1341}{3875} \approx 0.346$$
  - Áp dụng hàm mất mát `BCEWithLogitsLoss(pos_weight=pos_weight)` kết hợp với thuật toán tối ưu `Adam(lr=0.0005)`.
- **Bước 4 (Cơ chế suy luận & Dự đoán xác suất - Sigmoid):**
  - Mô hình trả về 1 Logit $z$. Xác suất bệnh nhân bị viêm phổi được tính qua hàm Sigmoid:
    $$P(\text{PNEUMONIA}) = \sigma(z) = \frac{1}{1 + e^{-z}} \times 100\%$$
    $$P(\text{NORMAL}) = 100\% - P(\text{PNEUMONIA})$$
  - Ngưỡng quyết định (Decision Threshold): $P \ge 50\% \rightarrow$ Viêm Phổi, ngược lại là Bình Thường.

---

## 4. Kết Quả Huấn Luyện & Đánh Giá Chi Tiết

### Bảng Tổng Hợp Chỉ Số Đánh Giá (Tập Kiểm Thử Độc Lập - 624 Ảnh)
Kết quả đo lường trên toàn bộ 624 ảnh phim X-quang của tập `test` (chưa từng xuất hiện trong quá trình huấn luyện):

| Chỉ Số Đánh Giá | Giá Trị Đạt Được | Ý Nghĩa Chuyên Môn |
| :--- | :---: | :--- |
| **Recall (Lớp Viêm Phổi)** | **`98.72%`** | **Tối quan trọng trong y tế:** Mô hình phát hiện được 385 / 390 ca viêm phổi thực tế, chỉ bỏ sót đúng 5 ca nguy kịch. |
| **Precision (Lớp Bình Thường)** | **`95.54%`** | Khi mô hình kết luận bệnh nhân có lá phổi khỏe mạnh thì độ tin cậy đạt tới 95.54%. |
| **F1-Score (Lớp Viêm Phổi)** | **`0.85`** | Cân bằng hài hòa giữa độ chính xác và độ phủ nhận diện mầm bệnh. |
| **Độ Chính Xác Tổng Thể (Accuracy)** | **`78.85%`** | Hiệu năng tổng thể vững chắc trên tập kiểm thử phân phối thực tế. |

---

### Chi Tiết Báo Cáo Phân Loại (Classification Report & Confusion Matrix)

```text
==================================================
KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST (624 ẢNH)
==================================================
Độ chính xác tổng thể (Accuracy): 0.7885 (78.85%)
Độ nhạy viêm phổi (Recall):       0.9872 (98.72%)
F1-Score viêm phổi:               0.8537 (0.85)
==================================================

Ma Trận Nhầm Lẫn (Confusion Matrix):
               Dự đoán NORMAL    Dự đoán PNEUMONIA
Thực tế NORMAL       107                127
Thực tế PNEUMONIA      5                385

Bảng Chi Tiết Chỉ Số (Classification Report):
                 precision    recall  f1-score   support

Bình thường (0)       0.96      0.46      0.62       234
  Viêm phổi (1)       0.75      0.99      0.85       390

       accuracy                           0.79       624
      macro avg       0.85      0.72      0.74       624
   weighted avg       0.83      0.79      0.77       624
```

---

### Kết Quả Thử Nghiệm Ngẫu Nhiên Trên 10 Ca Bệnh Thực Tế

Kiểm tra ngẫu nhiên trên 10 ca bệnh thuộc cả 2 nhóm trong tập Test Set:

```text
==========================================================================================
KIỂM TRA DỰ ĐOÁN X-QUANG TRÊN 10 CA THỰC TẾ
==========================================================================================
STT | Tên file ảnh      | Bệnh án thực tế | AI Dự đoán   | Xác suất Viêm phổi | Kết quả 
------------------------------------------------------------------------------------------
 1  | IM-0001-0001.jpeg | Bình thường     | Bình thường  |       12.85%       | Đúng
 2  | IM-0003-0001.jpeg | Bình thường     | Viêm phổi    |       96.59%       | Sai (Dương tính giả)
 3  | IM-0005-0001.jpeg | Bình thường     | Viêm phổi    |       80.63%       | Sai (Dương tính giả)
 4  | IM-0006-0001.jpeg | Bình thường     | Viêm phổi    |       94.84%       | Sai (Dương tính giả)
 5  | IM-0007-0001.jpeg | Bình thường     | Bình thường  |       14.04%       | Đúng
 6  | person100_bact... | Viêm phổi       | Viêm phổi    |       99.92%       | Đúng
 7  | person100_bact... | Viêm phổi       | Viêm phổi    |       99.52%       | Đúng
 8  | person100_bact... | Viêm phổi       | Viêm phổi    |      100.00%       | Đúng
 9  | person100_bact... | Viêm phổi       | Viêm phổi    |       99.98%       | Đúng
10  | person100_bact... | Viêm phổi       | Viêm phổi    |      100.00%       | Đúng
==========================================================================================
```

> **Chi tiết một ca bệnh viêm phổi nặng điển hình:**
> ```text
> ============================================================
> CHI TIẾT CA BỆNH - PNEUMONIA (MẪU SỐ 8)
> ============================================================
> Tên file:            person100_bacteria_478.jpeg
> Bệnh án thực tế:     Viêm phổi (PNEUMONIA)
> Chẩn đoán của AI:    Viêm phổi (PNEUMONIA)
> Xác suất Viêm phổi:  100.00%
> Xác suất Bình thường: 0.00%
> Trạng thái:          CHÍNH XÁC TUYỆT ĐỐI
> ============================================================
> ```

---

## 5. Đánh Giá & Kết Luận Chuyên Môn

1. **Ý nghĩa sống còn của chỉ số Recall trong chẩn đoán y tế:**
   - Trong lĩnh vực chẩn đoán hình ảnh y khoa, nguyên tắc hàng đầu là **"Thà nghi ngờ nhầm (False Positive) còn hơn bỏ sót mầm bệnh (False Negative)"**.
   - Mô hình đạt **Recall 98.72%** (chỉ có 5 / 390 ca viêm phổi bị phân loại nhầm thành bình thường). Điều này đảm bảo tính an toàn tối đa khi ứng dụng mô hình như một công cụ hỗ trợ sàng lọc sơ bộ (Pre-screening Tool) cho bác sĩ tại các bệnh viện tuyến dưới.
2. **Cơ chế hàm mất mát `BCEWithLogitsLoss` với `pos_weight`:**
   - Việc tích hợp sẵn hàm Sigmoid vào hàm BCE giúp ngăn chặn hiện tượng mất ổn định số học (Numerical Instability).
   - Hệ số `pos_weight = 0.346` kiểm soát mức phạt hợp lý đối với lớp viêm phổi chiếm đa số, giúp mô hình không bị thiên lệch nhãn một chiều.
3. **Sự kết hợp hoàn hảo giữa Mô hình AI và Giao diện Desktop:**
   - Ứng dụng Desktop cho phép tự động cập nhật lại trọng số mô hình khi có thêm nguồn dữ liệu X-quang mới mà không cần phải mở code hay can thiệp vào môi trường dòng lệnh phức tạp.

---

## 6. Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Cài đặt thư viện phụ thuộc
Mở Terminal tại thư mục dự án và chạy:

```bash
pip install -r requirements.txt
```

### 2. Khởi chạy Ứng dụng Desktop (Khuyến nghị)
Chạy ứng dụng với giao diện đồ họa cao cấp:

```bash
python du_doan_viem_phoi.py
```

- **Thao tác 1:** Bấm nút **"📁 Chọn Ảnh X-Quang Từ Máy Tính"** $\rightarrow$ Hệ thống tự động phân tích và đưa ra xác suất 2 màu Xanh / Đỏ tức thì.
- **Thao tác 2:** Khi có ảnh chụp mới, bạn chỉ cần copy ảnh vào `chest_xray/chest_xray/train/NORMAL` hoặc `PNEUMONIA`, sau đó bấm nút **"🔄 Huấn Luyện Lại (Re-Train)"** trên góc phải log để mô hình tự học lại ngầm.

### 3. Huấn luyện lại mô hình qua dòng lệnh (CLI Training)
Nếu muốn chạy huấn luyện trực tiếp qua script độc lập:

```bash
python src/train.py
```
*Trọng số mô hình tối ưu sẽ tự động được lưu đè vào thư mục `saved_models/best_chest_xray_model.pth`.*

---

## 7. Tác Giả & Bản Quyền
- **Tác giả:** Lê Hữu Hoàng (UNETI - Đại học Kinh tế - Kỹ thuật Công nghiệp)
- **GitHub:** [lhhoang5425](https://github.com/lhhoang5425)
