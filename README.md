# U²-Net: Phát Hiện Vật Thể Nổi Bật với Kiến Trúc U Lồng Nhau

**Bài báo:** [U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/pdf/2005.09007.pdf)  
**Tác giả:** [Xuebin Qin](https://xuebinqin.github.io/), [Zichen Zhang](https://webdocs.cs.ualberta.ca/~zichen2/), [Chenyang Huang](https://chenyangh.com/), [Masood Dehghan](https://sites.google.com/view/masooddehghan), [Osmar R. Zaiane](http://webdocs.cs.ualberta.ca/~zaiane/), [Martin Jagersand](https://webdocs.cs.ualberta.ca/~jag/)  
**Công bố:** Pattern Recognition 2020 &nbsp;|&nbsp; 🏆 Giải thưởng Bài báo hay nhất 2020

---
## Link data-set and kết quả của nhóm : https://drive.google.com/drive/folders/1TxRmD0v-zoxMwwgj8gddqSDhHUVEi0xq?fbclid=IwY2xjawSU9YFleHRuA2FlbQIxMABicmlkETFMbkQ5bjZSNlk1YnNzU3I2c3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MgABHub5TpdQ-1bJKRfMl0wTAj_60aU8c4COaDHyyG9JhXDpxVUTbtJDrPwzmgaS_aem_aCAVVCd-dBIMBgOHnLaEKg

## Giới thiệu

Dự án nghiên cứu bài toán **Salient Object Detection (SOD)** sử dụng kiến trúc U²-Net với khối **Residual U-block (RSU)** — cấu trúc "U trong U" cho phép trích xuất đặc trưng đa tỉ lệ mà không cần backbone tiền huấn luyện.

Đóng góp chính: đề xuất **Hybrid Loss (BCE + SSIM + IoU)** cải thiện MAE ↓30.8% và S-measure ↑0.97% so với BCE đơn thuần.

---

## Kết quả (DUTS-TE)

| Metric | BCE | Hybrid Loss | Δ |
|--------|:---:|:---:|:---:|
| MAE ↓ | 0.0801 | **0.0554** | ↓ 30.8% |
| maxFβ ↑ | 0.9355 | 0.9317 | ≈ |
| S-measure ↑ | 0.9161 | **0.9250** | ↑ +0.97% |

---


## Cài đặt & Sử dụng

```bash
git clone https://github.com/<your-username>/U-2_Net_Model.git
cd U-2_Net_Model
pip install -r requirements.txt
```

**Huấn luyện:**
```bash
python train.py --dataset DUTS --loss hybrid --epochs 40
```

**Inference:**
```bash
python inference.py --image path/to/image.jpg --checkpoint checkpoints/best.pth
```

---

## Cấu hình thực nghiệm

| Thông số | Giá trị |
|----------|---------|
| GPU | RTX 3050 Laptop (4.29 GB VRAM) |
| Batch size | 2 |
| Learning rate | 0.0001 + ReduceLROnPlateau |
| Early Stopping | patience = 7 |
| Input size | 256×256 |

---

## Nhóm thực hiện

**PBL4 — Khoa Điện Tử Viễn Thông, ĐH Bách khoa Đà Nẵng (2025–2026)**

| Thành viên | Email |
|------------|-------|
| Nguyễn Văn Quý | nquy1357@gmail.com |
| Hồ Nguyên Tâm | honguyentam57@gmail.com |
| Trần Đăng Bách | bach9747@gmail.com |

**GVHD:** TS. Hồ Phước Tiến
