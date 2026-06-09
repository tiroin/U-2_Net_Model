# U-2_Net_Model


# U²-Net: Đi Sâu Hơn với Cấu Trúc U Lồng Nhau để Phát Hiện Đối Tượng Nổi Bật

<p align="center">
  <img width="320" height="320" src="figures/U2Net_Logo.png">
</p>

**Bài báo:** [U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/pdf/2005.09007.pdf)

**Các tác giả:** [Xuebin Qin](https://xuebinqin.github.io/), [Zichen Zhang](https://webdocs.cs.ualberta.ca/~zichen2/), [Chenyang Huang](https://chenyangh.com/), [Masood Dehghan](https://sites.google.com/view/masooddehghan), [Osmar R. Zaiane](http://webdocs.cs.ualberta.ca/~zaiane/), [Martin Jagersand](https://webdocs.cs.ualberta.ca/~jag/)

**Công bố:** Pattern Recognition 2020

**Giải thưởng:** 🏆 **GIẢI THƯỞNG BÀI BÁO HAY NHẤT 2020 Pattern Recognition**

---

## 📑 Mục lục

1. [Tóm tắt](#tóm-tắt)
2. [Đóng góp chính](#đóng-góp-chính)
3. [Kiến trúc mô hình](#kiến-trúc-mô-hình)
4. [Cài đặt và yêu cầu](#cài-đặt-và-yêu-cầu)
5. [Cấu trúc dự án](#cấu-trúc-dự-án)
6. [Bắt đầu nhanh](#bắt-đầu-nhanh)
7. [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
8. [So sánh với các phương pháp khác](#so-sánh-với-các-phương-pháp-khác)
9. [Hàm mất mát](#hàm-mất-mát)
10. [Đào tạo mô hình](#đào-tạo-mô-hình)
11. [Suy luận](#suy-luận)
12. [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Tóm tắt

U²-Net là một kiến trúc học sâu mới cho phát hiện đối tượng nổi bật dựa trên cấu trúc U lồng nhau. Không giống như các phương pháp truyền thống dựa trên các tính năng thủ công hoặc các mô hình học sâu tiêu chuẩn, U²-Net tích hợp các cấu trúc mã hóa-giải mã lồng nhau với nhiều kích thước tiếp nhận trường để nắm bắt cả các tính năng nổi bật toàn cục và cục bộ ở các mức trừu tượng khác nhau.

### Các đặc điểm chính:

- **Kiến trúc Nested U-Structure**: Kết hợp các cấu trúc U ở các cấp độ khác nhau
- **Receptive Fields đa cấp**: Nắm bắt đặc trưng ở các tỷ lệ khác nhau
- **Mô hình nhẹ hiệu quả**: U2NETP với hiệu suất cao nhưng chi phí tính toán thấp
- **Hiệu suất tiên tiến**: Vượt trội trên các bộ dữ liệu chuẩn (ECSSD, DUT-OMRON, DUTS-TE, v.v.)

---

## Đóng góp chính

### 1. Kiến trúc Nested U-Structure
Bài báo đề xuất một cấu trúc U lồng nhau mới:
- Các khối mã hóa-giải mã được thay thế bằng các cấu trúc U nhỏ hơn
- Cho phép mô hình học các mô tả tính năng ở các mức trừu tượng khác nhau
- Cải thiện khả năng nắm bắt các chi tiết và bối cảnh toàn cục

### 2. Hàm Mất Mát Lai Ghép (Hybrid Loss)
- Kết hợp Binary Cross Entropy Loss ở tất cả các mức đầu ra
- Cho phép mô hình học được các biểu diễn tính năng phong phú ở các độ sâu khác nhau

### 3. Hiệu quả Tính Toán
- Mô hình U2NETP nhẹ hơn U2NET ~40 lần
- Vẫn duy trì độ chính xác cao trên các bộ dữ liệu tiêu chuẩn

### 4. Tính Linh Hoạt
- Có thể được tinh chỉnh cho các nhiệm vụ khác nhau (chân dung, phân đoạn tư nhân, v.v.)

---

## Kiến trúc Mô Hình

### Kiến trúc U²-Net Tổng Quan

```
Đầu vào (H×W×3)
    ↓
[En1 → De1] (Encode/Decode Block 1)
    ↓
[En2 → De2] (Encode/Decode Block 2)
    ↓
[En3 → De3] (Encode/Decode Block 3)
    ↓
[En4 → De4] (Encode/Decode Block 4)
    ↓
[En5 → De5] (Encode/Decode Block 5)
    ↓
[En6 → De6] (Encode/Decode Block 6)
    ↓
Concatenate + Convolution
    ↓
Đầu ra (H×W×1) - Bản đồ Saliency
```

### Nested U-Structure Chi Tiết

Mỗi khối mã hóa/giải mã chứa:
- **RSU (Residual U-block)**: Các khối U nhỏ được xếp chồng lên nhau
- **Phép cộng Residual**: Giúp training của các mạng sâu
- **Dilated Convolution**: Mở rộng trường tiếp nhận

### Kiến trúc Chi Tiết

```python
U²-Net Architecture:
├─ Encoder Stage 1-6
│  └─ RSU-4F ~ RSU-7 (Residual U-block)
│     ├─ Convolution 3×3
│     ├─ Max Pooling
│     ├─ Nested U-block
│     ├─ Upsampling
│     └─ Concatenate
├─ Decoder Stage 1-6
│  └─ RSU-4 ~ RSU-5
├─ Output Fusion
└─ Sigmoid Activation
```

---

## Cài đặt và yêu cầu

### Yêu cầu Hệ Thống

- Python 3.6+
- PyTorch >= 1.0
- CUDA 10.0+ (nếu dùng GPU)
- GPU Memory: 8GB+ (được khuyến nghị)

### Cài đặt Phần Mềm

```bash
# Clone kho lưu trữ
git clone https://github.com/xuebinqin/U-2-Net.git
cd U-2-Net

# Tạo môi trường ảo (tùy chọn nhưng được khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt các phụ thuộc
pip install -r requirements.txt
```

### Các Gói Bắt Buộc

```
torch>=1.0.0              # PyTorch framework
torchvision>=0.4.0        # Computer vision utilities
opencv-python>=4.0        # Image processing
scikit-image>=0.14.0      # Image processing utilities
numpy>=1.15.0             # Numerical computations
pillow>=8.0.0             # Image manipulation
gradio                    # Web interface
```
│   └── test_human_images/          # Hình ảnh kiểm tra phân đoạn con người
│
├── gradio/                         # Giao diện demo Gradio
│   └── demo.py
│
├── data_loader.py                  # Tiện ích tải tập dữ liệu
├── u2net_train.py                  # Tập lệnh đào tạo
├── u2net_test.py                   # Tập lệnh kiểm tra
├── u2net_portrait_demo.py          # Demo phân đoạn chân dung
├── u2net_portrait_test.py          # Kiểm tra phân đoạn chân dung
├── u2net_human_seg_test.py         # Kiểm tra phân đoạn con người
├── u2net_portrait_composite.py     # Tiện ích sáng tác chân dung
│
└── requirements.txt                # Phụ thuộc Python
```

## Bắt đầu nhanh

### 1. Kiểm tra với mô hình được đào tạo trước

Kiểm tra trên các hình ảnh mẫu:
```bash
python u2net_test.py
```

### 2. Demo phân đoạn chân dung

Chạy demo phân đoạn chân dung với phát hiện khuôn mặt:
```bash
python u2net_portrait_demo.py
```

### 3. Phân đoạn con người

Phát hiện và phân đoạn con người trong hình ảnh:
```bash
python u2net_human_seg_test.py
```

### 4. Giao diện web tương tác

Khởi chạy demo Gradio để kiểm tra dễ dàng:
```bash
cd gradio
python demo.py
```

Sau đó mở trình duyệt của bạn đến URL cục bộ (thường là `http://localhost:7860`)

## Mô hình và trọng số được đào tạo trước

### Các mô hình có sẵn

| Mô hình | Mục đích | Kích thước | Hiệu suất |
|---------|---------|-----------|----------|
| **U2NET** | Phát hiện đối tượng nổi bật chung | Lớn | Độ chính xác tốt nhất |
| **U2NETP** | Biến thể nhẹ | Nhỏ | Suy luận nhanh |
| **Portrait** | Phân đoạn chân dung/khuôn mặt | Trung bình | Được tối ưu hóa cho khuôn mặt |

### Vị trí trọng số được đào tạo trước

```
saved_models/
├── u2net/u2net_best.pth           - Mô hình chung tốt nhất
├── u2net_portrait/u2net.pth        - Phân đoạn chân dung
├── cuaho/u2net.pth                 - Mô hình được đào tạo thay thế
├── ecssd.pth                       - Mô hình bộ dữ liệu ECSSD
├── dut-tr.pth                      - Mô hình bộ dữ liệu DUT-TR
└── [various_hybrid_models].pth     - Các mô hình được đào tạo mất mát lai ghép
```

## Cấu trúc dự án

```
U-2-Net/
├── model/
│   ├── u2net.py                 # Triển khai mô hình U2NET
│   ├── u2net_refactor.py        # Phiên bản tái cấu trúc
│   └── __init__.py
│
├── saved_models/                # Trọng số được đào tạo trước
│   ├── u2net/                   # Mô hình U2NET
│   ├── u2net_portrait/          # Mô hình phân đoạn chân dung
│   ├── cuaho/                   # Mô hình tùy chỉnh
│   └── test_train_1/            # Kết quả đào tạo thử nghiệm
│
├── train_data/DUTS/             # Tập dữ liệu đào tạo
├── test_data/                   # Tập dữ liệu kiểm tra
├── gradio/demo.py              # Giao diện web Gradio
│
├── data_loader.py              # Tải và xử lý dữ liệu
├── u2net_train.py              # Tập lệnh đào tạo
├── u2net_test.py               # Tập lệnh kiểm tra
├── u2net_portrait_demo.py      # Demo phân đoạn chân dung
├── u2net_human_seg_test.py     # Kiểm tra phân đoạn con người
│
└── requirements.txt            # Phụ thuộc

---

## Bắt đầu nhanh

### 1. Kiểm tra mô hình được đào tạo trước

```bash
python u2net_test.py
# Đầu vào: test_data/test_images/
# Đầu ra: test_data/u2net_results/
```

### 2. Phân đoạn chân dung với phát hiện khuôn mặt

```bash
python u2net_portrait_demo.py
# Xử lý các hình ảnh chân dung với phát hiện tự động khuôn mặt
```

### 3. Phân đoạn con người

```bash
python u2net_human_seg_test.py
# Đầu vào: test_data/test_human_images/
# Đầu ra: test_data/u2net_test_human_images_results/
```

### 4. Giao diện web tương tác

```bash
cd gradio
python demo.py
# Truy cập: http://localhost:7860
```

---

## Kết quả thực nghiệm

### Bộ dữ liệu Kiểm tra

U²-Net được đánh giá trên 5 bộ dữ liệu chuẩn:

| Bộ dữ liệu | Ảnh | Kích thước | Đặc điểm |
|-----------|------|---------|----------|
| **ECSSD** | 1000 | Phức tạp, cạnh khó | Bảo toàn cạnh |
| **DUT-OMRON** | 5172 | Đối tượng cạnh tranh | Phát hiện mạnh |
| **DUTS-TE** | 5019 | Đa dạng, quy mô lớn | Khát tổng quát |
| **HKU-IS** | 4447 | Đối tượng chồng chéo | Xử lý chồng chéo |
| **SED2** | 100 | Ảnh lớn, cạnh khó | Ảnh lớn |

### Chỉ số Đánh giá

Bài báo sử dụng ba chỉ số đánh giá chính:

1. **MAE (Mean Absolute Error)** - Sai số trung bình tuyệt đối
   - Thấp hơn tốt hơn
   
2. **Fβ Score** - Điểm F được trộng lại
   - Cao hơn tốt hơn
   - Cân bằng giữa Precision và Recall
   
3. **S-measure** - Đo độ tương tự cấu trúc
   - Cao hơn tốt hơn
   - Xem xét cấu trúc đối tượng

### Kết quả Định lượng

#### U2NET - Kết quả Tốt Nhất

```
Bộ dữ liệu    | MAE ↓  | Fβ ↑  | S-measure ↑
-------------|--------|-------|-------------
ECSSD       | 0.037  | 0.911 | 0.928
DUT-OMRON   | 0.058  | 0.797 | 0.863
DUTS-TE     | 0.062  | 0.799 | 0.855
HKU-IS      | 0.052  | 0.896 | 0.914
SED2        | 0.077  | 0.760 | 0.805
```

#### U2NETP - Mô hình nhẹ (40× nhỏ hơn)

```
Bộ dữ liệu    | MAE ↓  | Fβ ↑  | S-measure ↑
-------------|--------|-------|-------------
ECSSD       | 0.049  | 0.887 | 0.903
DUT-OMRON   | 0.068  | 0.758 | 0.827
DUTS-TE     | 0.073  | 0.770 | 0.822
```

---

## So sánh với các phương pháp khác

### Bộ so sánh ECSSD (SOTA - State-of-the-Art)

| Phương pháp | Năm | MAE ↓ | Fβ ↑ | Ghi chú |
|------------|-----|-------|-------|---------|
| **U2NET** ⭐ | **2020** | **0.037** | **0.911** | **Bài báo này** |
| PoolNet | 2019 | 0.041 | 0.909 | Trước U2NET |
| EGNet | 2019 | 0.039 | 0.910 | Cạnh tranh |
| CPD | 2019 | 0.047 | 0.887 | Cấp độ cao |
| CapSalNet | 2019 | 0.048 | 0.887 | Cấp độ cao |

### Bộ so sánh DUT-OMRON

| Phương pháp | MAE ↓ | Fβ ↑ |
|------------|-------|-------|
| **U2NET** | **0.058** | **0.797** |
| PoolNet | 0.063 | 0.789 |
| CPD | 0.062 | 0.795 |
| EGNet | 0.061 | 0.795 |

---

## Hàm Mất Mát

### Binary Cross Entropy Loss

Hàm mất mát chính được sử dụng:

$$L(S, G) = -\frac{1}{N}\sum_{i=1}^{N}[G_i \log S_i + (1-G_i)\log(1-S_i)]$$

Trong đó:
- $S_i$ - Xác suất dự đoán ở vị trí $i$
- $G_i$ - Nhãn sự thật tại vị trí $i$
- $N$ - Tổng số pixel

### Hàm Mất Mát Lai Ghép (Hybrid Loss)

Bài báo đề xuất kết hợp mất mát từ tất cả các mức đầu ra:

$$L_{total} = \sum_{i=0}^{6} L(S_i, G)$$

Trong đó:
- $S_i$ - Bản đồ saliency ở mức $i$ (từ 0 đến 6)
- Mức 0 là đầu ra cuối cùng (độ phân giải cao)
- Mức 6 là đầu ra ban đầu (độ phân giải thấp)

### Lợi ích của Hybrid Loss

1. **Học sâu**: Giúp gradient lan truyền đến các lớp sâu
2. **Học đa tỷ lệ**: Nắm bắt đặc trưng ở các tỷ lệ khác nhau
3. **Tổng hợp đặc trưng**: Khuyến khích sự kết hợp tốt của các đặc trưng
4. **Giải quyết vanishing gradient**: Hạn chế vấn đề gradient biến mất

---

## Đào tạo Mô hình

### Chuẩn bị Dữ liệu

Định dạng dữ liệu đào tạo (DUTS-TR):

```
train_data/DUTS/DUTS-TR/DUTS-TR/
├── im_aug/              # Ảnh đào tạo
│   ├── ILSVRC_train_00000001.jpg
│   ├── ILSVRC_train_00000002.jpg
│   └── ...
└── gt_aug/              # Mặt nạ sự thật
    ├── ILSVRC_train_00000001.png
    ├── ILSVRC_train_00000002.png
    └── ...
```

### Cấu hình Đào tạo

Tham số chính trong `u2net_train.py`:

```python
# Cấu hình mô hình
model_name = 'u2net'            # hoặc 'u2netp'
num_channels = 3                # RGB

# Cấu hình dữ liệu
batch_size_train = 12           # Kích thước batch
batch_size_val = 1              # Batch xác thực
epoch_num = 2000                # Số epoch

# Cấu hình tối ưu hóa
learning_rate = 1e-3            # Tỷ lệ học
betas = (0.9, 0.999)            # Adam optimizer
weight_decay = 0                # Quy chuẩn L2
```

### Quy trình Đào tạo

```bash
# Đào tạo từ đầu
python u2net_train.py

# Hoặc sử dụng Jupyter Notebook
jupyter notebook U2Net_Colab_Train.ipynb
```

### Giám sát Đào tạo

Tập lệnh đào tạo sẽ:

1. **Lưu checkpoint** - Mỗi epoch
2. **In nhật ký** - Mất mát ở tất cả các mức
3. **Xác thực** - Sau mỗi epoch
4. **Lưu mô hình tốt nhất** - Dựa trên hiệu suất validation

Nhật ký đào tạo mẫu:
```
Epoch 1 [1/1000], Training Loss = 0.1234
  l0: 0.0123, l1: 0.0234, l2: 0.0345, l3: 0.0456, ...
Epoch 1: Validation mIoU = 0.6234
```

---

## Suy Luận

### Cách sử dụng mô hình đơn giản

```python
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import U2NET

# Load mô hình
model = U2NET(3, 1)
model.load_state_dict(torch.load('saved_models/u2net/u2net_best.pth'))
model.cuda()  # hoặc .cpu()
model.eval()

# Chuẩn bị ảnh
image = Image.open('test.jpg').convert('RGB')
image_tensor = transforms.ToTensor()(image).unsqueeze(0)
image_tensor = image_tensor.cuda()  # hoặc .cpu()

# Suy luận
with torch.no_grad():
    outputs = model(image_tensor)
    output = outputs[0]  # Lấy đầu ra cuối cùng
    output = F.sigmoid(output)  # Áp dụng sigmoid
    output = output.cpu().data.numpy()
    
# Chuẩn hóa về [0, 255]
output = (output * 255).astype('uint8').squeeze()

# Lưu kết quả
Image.fromarray(output).save('result.png')
```

### Xử lý hàng loạt

```python
import os
import glob

test_dir = 'test_data/test_images/'
output_dir = 'test_data/u2net_results/'

for img_path in glob.glob(os.path.join(test_dir, '*.jpg')):
    # Xử lý như trên
    # Lưu kết quả vào output_dir
    pass
```

### Các lớp đầu ra

Mô hình U²-Net tạo ra 7 đầu ra ở độ phân giải khác nhau:

```python
d0, d1, d2, d3, d4, d5, d6 = model(input_tensor)
# d0: Độ phân giải gốc (H × W × 1)
# d1-d6: Độ phân giải giảm dần
```

---

## Tài liệu Tham khảo

### Bài báo Gốc

```bibtex
@article{Qin2020U2Net,
  title={U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
  author={Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar R and Jagersand, Martin},
  journal={Pattern Recognition},
  volume={106},
  pages={107404},
  year={2020},
  publisher={Elsevier}
}
```

### Các liên kết quan trọng

- **Bài báo ArXiv**: https://arxiv.org/abs/2005.09007
- **Mã nguồn chính thức**: https://github.com/NathanUA/U-2-Net
- **Tác giả chính**: [Xuebin Qin](https://xuebinqin.github.io/)

### Bộ dữ liệu Được sử dụng

1. **DUTS-TR** - Đào tạo (10582 ảnh)
2. **ECSSD** - Kiểm tra (1000 ảnh)
3. **DUT-OMRON** - Kiểm tra (5172 ảnh)
4. **DUTS-TE** - Kiểm tra (5019 ảnh)
5. **HKU-IS** - Kiểm tra (4447 ảnh)
6. **SED2** - Kiểm tra (100 ảnh)

---

## Ứng dụng và Mở rộng

### Các ứng dụng đã biến thành sản phẩm

1. **Rembg** - Xóa nền ảnh (https://github.com/danielgatis/rembg)
2. **Silueta** - Webapp xóa nền (https://silueta.me/)
3. **Hotpot.ai** - Công cụ thiết kế AI
4. **Ứng dụng iOS** - Portrait Drawing, 3D Photo Creator, Lensto

### Những mở rộng có thể

- Phân đoạn chân dung chi tiết
- Phát hiện từng bộ phận cơ thể
- Phân đoạn quần áo
- Phân đoạn bầu trời
- Video saliency detection

---

## Giải quyết vấn đề

### CUDA/GPU

```bash
# Kiểm tra GPU
python -c "import torch; print(torch.cuda.is_available())"

# Chạy trên CPU
python u2net_test.py  # Sẽ tự động chọn CPU nếu không có GPU
```

### Lỗi tải mô hình

```bash
# Đảm bảo tệp trọng số tồn tại và hoàn chỉnh
python -c "import torch; torch.load('saved_models/u2net/u2net_best.pth')"
```

### Hết bộ nhớ (OOM)

1. Giảm `batch_size_train`
2. Sử dụng U2NETP thay vì U2NET
3. Giảm kích thước ảnh đầu vào
4. Xóa các ảnh lớn khỏi tập dữ liệu

---

## Liên hệ và Hỗ trợ

- **Email tác giả**: xuebin[at]ualberta[dot]ca
- **GitHub Issues**: Tạo issue cho các câu hỏi và báo cáo lỗi
- **Discussions**: Thảo luận về các ứng dụng và mở rộng

---

## Giấy Phép

Xem file [LICENSE](LICENSE) để biết chi tiết về giấy phép.

## Lời Cảm ơn

- Các tác giả chính của bài báo
- Cộng đồng nghiên cứu và phát triển
- Các nhà cung cấp bộ dữ liệu (DUTS, ECSSD, DUT-OMRON, HKU-IS, SED2)
- Tất cả những ai đã đóng góp và sử dụng U²-Net

---

**Cập nhật lần cuối**: 2024
**Giải thưởng**: 🏆 Pattern Recognition Best Paper Award 2020

![Sample Results: Ladies](figures/portrait_ladies.png)

![Sample Results: Men](figures/portrait_men.png)

### Usage for portrait generation
1. Clone this repo to local
```
git clone https://github.com/NathanUA/U-2-Net.git
```

2. Download the u2net_portrait.pth from [**GoogleDrive**](https://drive.google.com/file/d/1IG3HdpcRiDoWNookbncQjeaPN28t90yW/view?usp=sharing) or [**Baidu Pan(提取码：chgd)**](https://pan.baidu.com/s/1BYT5Ts6BxwpB8_l2sAyCkw)model and put it into the directory: ```./saved_models/u2net_portrait/```.

3. Run on the testing set. <br/>
(1) Download the train and test set from [**APDrawingGAN**](https://github.com/yiranran/APDrawingGAN). These images and their ground truth are stitched side-by-side (512x1024). You need to split each of these images into two 512x512 images and put them into ```./test_data/test_portrait_images/portrait_im/```. You can also download the split testing set on [GoogleDrive](https://drive.google.com/file/d/1NkTsDDN8VO-JVik6VxXyV-3l2eo29KCk/view?usp=sharing). <br/>
(2) Running the inference with command ```python u2net_portrait_test.py``` will ouptut the results into ```./test_data/test_portrait_images/portrait_results```. <br/>

4. Run on your own dataset. <br/>
(1) Prepare your images and put them into ```./test_data/test_portrait_images/your_portrait_im/```. [**To obtain enough details of the protrait, human head region in the input image should be close to or larger than 512x512. The head background should be relatively clear.**](https://github.com/NathanUA/U-2-Net) <br/>
(2) Run the prediction by command ```python u2net_portrait_demo.py``` will outputs the results to ```./test_data/test_portrait_images/your_portrait_results/```. <br/>
(3) The difference between ```python u2net_portrait_demo.py``` and ```python u2net_portrait_test.py``` is that we added a simple [**face detection**](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html) step before the portrait generation in ```u2net_portrait_demo.py```.  Because the testing set of APDrawingGAN are normalized and cropped to 512x512 for including only heads of humans, while our own dataset may varies with different resolutions and contents. Therefore, the code ```python u2net_portrait_demo.py``` will detect the biggest face from the given image and then crop, pad and resize the ROI to 512x512 for feeding to the network. The following figure shows how to take your own photos for generating high quality portraits.

**(2020-Sep-13)** Our U<sup>2</sup>-Net based model is the **6th** in [**MICCAI 2020 Thyroid Nodule Segmentation Challenge**](https://tn-scui2020.grand-challenge.org/Resultannouncement/).

**(2020-May-18)** The official paper of our **U<sup>2</sup>-Net (U square net)** ([**PDF in elsevier**(free until July 5 2020)](https://www.sciencedirect.com/science/article/pii/S0031320320302077?dgcid=author), [**PDF in arxiv**](http://arxiv.org/abs/2005.09007)) is now available. If you are not able to access that, please feel free to drop me an email.

**(2020-May-16)** We fixed the upsampling issue of the network. Now, the model should be able to handle **arbitrary input size**. (Tips: This modification is to facilitate the retraining of U<sup>2</sup>-Net on your own datasets. When using our pre-trained model on SOD datasets, please keep the input size as 320x320 to guarantee the performance.)

**(2020-May-16)** We highly appreciate **Cyril Diagne** for building this fantastic AR project: [**AR Copy and Paste**](https://github.com/cyrildiagne/ar-cutpaste) using our **U<sup>2</sup>-Net** (Qin *et al*, PR 2020) and [**BASNet**](https://github.com/NathanUA/BASNet)(Qin *et al*, CVPR 2019). The [**demo video**](https://twitter.com/cyrildiagne/status/1256916982764646402) in twitter has achieved over **5M** views, which is phenomenal and shows us more application possibilities of SOD.

## U<sup>2</sup>-Net Results (176.3 MB)

![U<sup>2</sup>-Net Results](figures/u2netqual.png)


## Our previous work: [BASNet (CVPR 2019)](https://github.com/NathanUA/BASNet)

## Required libraries

Python 3.6  
numpy 1.15.2  
scikit-image 0.14.0  
python-opencv
PIL 5.2.0  
PyTorch 0.4.0  
torchvision 0.2.1  
glob  

## Usage for salient object detection
1. Clone this repo
```
git clone https://github.com/NathanUA/U-2-Net.git
```
2. Download the pre-trained model u2net.pth (176.3 MB) from [**GoogleDrive**](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing) or [**Baidu Pan 提取码: pf9k**](https://pan.baidu.com/s/1WjwyEwDiaUjBbx_QxcXBwQ) or u2netp.pth (4.7 MB) from [**GoogleDrive**](https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view?usp=sharing) or [**Baidu Pan 提取码: 8xsi**](https://pan.baidu.com/s/10tW12OlecRpE696z8FxdNQ) and put it into the dirctory './saved_models/u2net/' and './saved_models/u2netp/'

3.  Cd to the directory 'U-2-Net', run the train or inference process by command: ```python u2net_train.py```
or ```python u2net_test.py``` respectively. The 'model_name' in both files can be changed to 'u2net' or 'u2netp' for using different models.  

 We also provide the predicted saliency maps ([u2net results](https://drive.google.com/file/d/1mZFWlS4WygWh1eVI8vK2Ad9LrPq4Hp5v/view?usp=sharing),[u2netp results](https://drive.google.com/file/d/1j2pU7vyhOO30C2S_FJuRdmAmMt3-xmjD/view?usp=sharing)) for datasets SOD, ECSSD, DUT-OMRON, PASCAL-S, HKU-IS and DUTS-TE.


## U<sup>2</sup>-Net Architecture

![U<sup>2</sup>-Net architecture](figures/U2NETPR.png)


## Quantitative Comparison

![Quantitative Comparison](figures/quan_1.png)

![Quantitative Comparison](figures/quan_2.png)


## Qualitative Comparison

![Qualitative Comparison](figures/qual.png?raw=true)

## Citation
```
@InProceedings{Qin_2020_PR,
title = {U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
journal = {Pattern Recognition},
volume = {106},
pages = {107404},
year = {2020}
}
```
# U-2_Net_Mode2_2
