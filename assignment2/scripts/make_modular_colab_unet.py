import json
from pathlib import Path

cells = []

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

# -------------------------------------------------------------
# Header
# -------------------------------------------------------------
cells.append(md_cell("""# Assignment 2 — U-Net Baseline Runner (Git-Synced & Modular)
### Huấn luyện U-Net Scratch và Đánh Giá So Sánh Đối Đầu với DeepLabV3 (E0)
**Tác giả:** Cao Đức Thành (Nhóm RTX5090)  
**Khóa học:** CO3133 Deep Learning and Its Applications — HK261  
**Repository:** [CO3133_sem261_groupRTX5090](https://github.com/phsuuuuuu/CO3133_sem261_groupRTX5090)

---
### 🌟 Đặc điểm nổi bật của Notebook này:
1. **Chạy trực tiếp từ mã nguồn GitHub**: Clone repo từ GitHub và gọi trực tiếp các module Python (`scripts/prepare_data.py`, `scripts/create_splits.py`, `src/train.py`, `src/evaluate.py`) thay vì nhúng code vào notebook. Giúp kiểm tra chính xác tính đúng đắn của code trong repo.
2. **Hỗ trợ tối ưu phần cứng thế hệ mới G4 (NVIDIA RTX PRO 6000 Blackwell 96 GB GDDR7)**: Tự động nâng batch size = 32 và tối ưu workers, huấn luyện 40 epoch chỉ trong ~5-7 phút.
3. **Tự động nhận diện GPU Colab**: A100 (batch 16), L4 (batch 12), T4 (batch 8).
4. **Pre-resize siêu tốc 512x1024**: Giảm 4x thời gian đọc đĩa và loại bỏ 100% nghẽn CPU.
5. **So sánh đối đầu E0 tự động**: Trực tiếp đối chiếu kết quả của U-Net với 3 model DeepLabV3 đã có sẵn kết quả test trong repo."""))

# -------------------------------------------------------------
# Cell 1: Unified Config Panel
# -------------------------------------------------------------
cells.append(code_cell("""# @title 🎛️ BẢNG CẤU HÌNH TỔNG HỢP / UNIFIED CONFIGURATION PANEL
# @markdown Điền thông tin tại đây trước khi bấm Run All.

# @markdown ---
# @markdown ### 🌐 1. TÀI KHOẢN TẢI DATASET TỪ CITYSCAPES WEB
# @markdown Đăng ký miễn phí tại: https://www.cityscapes-dataset.com/register/
CITYSCAPES_USERNAME = ""  # @param {type:"string"}
CITYSCAPES_PASSWORD = ""  # @param {type:"string"}

# @markdown Hoặc nếu bạn đã có dataset trong Google Drive, tick chọn bên dưới:
USE_GOOGLE_DRIVE = False  # @param {type:"boolean"}
DRIVE_ZIP_FOLDER = "MyDrive/Cityscapes"  # @param {type:"string"}

# @markdown ---
# @markdown ### ⚡ 2. TỐI ƯU HÓA SIÊU TỐC PRE-RESIZE
FAST_PRE_RESIZE = True  # @param {type:"boolean"}
# @markdown Tự động pre-resize toàn bộ dataset về 512x1024 đa tiến trình 1 lần duy nhất lúc tải xong (tăng tốc train 4-5x).

# @markdown ---
# @markdown ### ⚙️ 3. HYPERPARAMETERS HUẤN LUYỆN U-NET
EPOCHS = 40  # @param {type:"integer"}
RANDOM_SEED = 666  # @param {type:"integer"}
UNET_LR = 0.001  # @param {type:"number"}

# @markdown Chế độ Batch Size:
# @markdown - 'Auto (Detect GPU)': Tự động đo VRAM (G4 RTX 6000 -> 32; A100 -> 16; L4 -> 12; T4 -> 8).
# @markdown - 'Match Local (8)': Chọn batch 8 y hệt máy RTX 5070 Ti local.
# @markdown - 'Custom': Dùng batch size và số workers tùy chỉnh ở 2 ô bên dưới.
BATCH_SIZE_MODE = "Auto (Detect GPU)"  # @param ["Auto (Detect GPU)", "Match Local (8)", "Custom"]
CUSTOM_BATCH_SIZE = 8  # @param {type:"integer"}
CUSTOM_NUM_WORKERS = 4  # @param {type:"integer"}
USE_AMP = True  # @param {type:"boolean"}

# @markdown Dry-run test nhanh (Đặt 0 để huấn luyện toàn bộ dataset; đặt > 0 ví dụ 20 để test nhanh pipeline):
MAX_TRAIN_BATCHES = 0  # @param {type:"integer"}
MAX_EVAL_BATCHES = 0  # @param {type:"integer"}

print("=" * 75)
print("🎛️  THÔNG TIN CẤU HÌNH U-NET ĐÃ THIẾT LẬP:")
print(f"   • Tải dữ liệu từ        : {'Google Drive (' + DRIVE_ZIP_FOLDER + ')' if USE_GOOGLE_DRIVE else ('Cityscapes Web (' + (CITYSCAPES_USERNAME or 'CHƯA ĐIỀN') + ')')}")
print(f"   • Siêu Tốc Pre-Resize   : {'BẬT (Khuyên dùng)' if FAST_PRE_RESIZE else 'TẮT'}")
print(f"   • Số Epochs             : {EPOCHS}")
print(f"   • Tốc độ học (AdamW LR) : {UNET_LR}")
print(f"   • Chế độ Batch Size     : {BATCH_SIZE_MODE} (Custom={CUSTOM_BATCH_SIZE})")
print(f"   • Mixed Precision (AMP) : {USE_AMP}")
print(f"   • Random Seed           : {RANDOM_SEED}")
print(f"   • Giới hạn Batches      : {'Toàn bộ Dataset' if MAX_TRAIN_BATCHES == 0 else f'{MAX_TRAIN_BATCHES} batches (Dry-run)'}")
print("=" * 75)"""))

# -------------------------------------------------------------
# Bước 1: Kiểm tra GPU
# -------------------------------------------------------------
cells.append(md_cell("## Bước 1: Kiểm Tra Tài Nguyên Phần Cứng GPU"))
cells.append(code_cell("""# @title Kiểm Tra Thông Số GPU
!nvidia-smi

import torch
print(f"\\nPyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name        : {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("⚠️ CẢNH BÁO: Chưa bật GPU! Hãy vào Runtime -> Change runtime type -> Chọn T4 hoặc A100.")"""))

# -------------------------------------------------------------
# Bước 2: Clone / Cập nhật Repository từ GitHub
# -------------------------------------------------------------
cells.append(md_cell("""## Bước 2: Clone / Cập Nhật Repository & Cài Đặt Gói Cần Thiết
Notebook sẽ clone trực tiếp mã nguồn dự án từ GitHub về môi trường Colab để sử dụng trực tiếp các file trong `src/` và `scripts/`."""))
cells.append(code_cell("""# @title Clone Repository từ GitHub và Cài Đặt Thư Viện
import os

REPO_URL = "https://github.com/phsuuuuuu/CO3133_sem261_groupRTX5090.git"
ROOT_DIR = "/content/CO3133_sem261_groupRTX5090"
ASSIGNMENT_DIR = f"{ROOT_DIR}/assignment2"

if not os.path.exists(ROOT_DIR):
    print(f"Cloning {REPO_URL}...")
    !git clone {REPO_URL} {ROOT_DIR}
else:
    print("Repository đã tồn tại. Đang đồng bộ commit mới nhất từ nhánh main...")
    !git -C {ROOT_DIR} checkout main
    !git -C {ROOT_DIR} pull origin main

%cd {ASSIGNMENT_DIR}

# Cài đặt các thư viện cần thiết
!pip install -q cityscapesscripts torchmetrics PyYAML tqdm appdirs requests
print("\\n✅ Môi trường và mã nguồn dự án đã sẵn sàng tại:", ASSIGNMENT_DIR)"""))

# -------------------------------------------------------------
# Bước 3: Tải Dataset
# -------------------------------------------------------------
cells.append(md_cell("""## Bước 3: Tải Dataset Cityscapes & Giải Nén
Tự động tải 2 gói `gtFine_trainvaltest.zip` và `leftImg8bit_trainvaltest.zip` từ web Cityscapes hoặc copy từ Google Drive."""))
cells.append(code_cell("""# @title Tải Dataset & Giải Nén Vào SSD Colab
import os
import sys
import json
import shutil
import appdirs
from pathlib import Path
from cityscapesscripts.download import downloader

LOCAL_DATASET = Path(ASSIGNMENT_DIR) / "dataset"
LOCAL_DATASET.mkdir(parents=True, exist_ok=True)

already_ready = (
    (LOCAL_DATASET / "leftImg8bit_trainvaltest" / "leftImg8bit").is_dir() and
    (LOCAL_DATASET / "gtFine_trainvaltest" / "gtFine").is_dir()
)

if already_ready:
    print("✅ Dataset đã tồn tại đầy đủ trên Colab. Bỏ qua bước tải về!")
elif USE_GOOGLE_DRIVE:
    print("📦 Đang mount Google Drive...")
    from google.colab import drive
    drive.mount("/content/drive")
    drive_path = Path("/content/drive") / DRIVE_ZIP_FOLDER
    print(f"Đang copy file zip từ {drive_path}...")
    for zip_name in ["gtFine_trainvaltest.zip", "leftImg8bit_trainvaltest.zip"]:
        src_zip = drive_path / zip_name
        if src_zip.is_file():
            print(f"  • Copying {zip_name}...")
            shutil.copy2(src_zip, LOCAL_DATASET / zip_name)
else:
    if not (CITYSCAPES_USERNAME and CITYSCAPES_PASSWORD):
        raise ValueError(
            "❌ Vui lòng điền CITYSCAPES_USERNAME và CITYSCAPES_PASSWORD ở Bảng cấu hình trên cùng!"
        )

    print(f"🔑 Đang đăng nhập Cityscapes portal: {CITYSCAPES_USERNAME}...")
    cred_dir = appdirs.user_data_dir("cityscapesscripts", "cityscapes")
    os.makedirs(cred_dir, exist_ok=True)
    with open(os.path.join(cred_dir, "credentials.json"), "w", encoding="utf-8") as f:
        json.dump({"username": CITYSCAPES_USERNAME, "password": CITYSCAPES_PASSWORD}, f)

    session = downloader.login()
    print("✅ Đăng nhập Cityscapes thành công!")

    packages = []
    if not (LOCAL_DATASET / "gtFine_trainvaltest.zip").exists() and not (LOCAL_DATASET / "gtFine_trainvaltest").is_dir():
        packages.append("gtFine_trainvaltest.zip")
    if not (LOCAL_DATASET / "leftImg8bit_trainvaltest.zip").exists() and not (LOCAL_DATASET / "leftImg8bit_trainvaltest").is_dir():
        packages.append("leftImg8bit_trainvaltest.zip")

    if packages:
        print(f"🚀 Đang tải trực tiếp: {packages}...")
        downloader.download_packages(session, packages, destination_path=str(LOCAL_DATASET), resume=True)

# Giải nén
print("\\n📦 Đang giải nén dataset vào SSD Colab...")
if not (LOCAL_DATASET / "gtFine_trainvaltest" / "gtFine").is_dir():
    if (LOCAL_DATASET / "gtFine_trainvaltest.zip").is_file():
        print("  • Giải nén gtFine_trainvaltest.zip (252 MB)...")
        !unzip -q -o "{LOCAL_DATASET}/gtFine_trainvaltest.zip" -d "{LOCAL_DATASET}/gtFine_trainvaltest"

if not (LOCAL_DATASET / "leftImg8bit_trainvaltest" / "leftImg8bit").is_dir():
    if (LOCAL_DATASET / "leftImg8bit_trainvaltest.zip").is_file():
        print("  • Giải nén leftImg8bit_trainvaltest.zip (11.5 GB)...")
        !unzip -q -o "{LOCAL_DATASET}/leftImg8bit_trainvaltest.zip" -d "{LOCAL_DATASET}/leftImg8bit_trainvaltest"

print("✅ Chuẩn bị dữ liệu thô hoàn tất!")"""))

# -------------------------------------------------------------
# Bước 4: Tạo trainIds & Splits bằng scripts thật
# -------------------------------------------------------------
cells.append(md_cell("""## Bước 4: Sinh trainIds Masks (34 $\\rightarrow$ 19 nhãn) & Chia Splits
Gọi trực tiếp `scripts/prepare_data.py` và `scripts/create_splits.py` từ kho mã nguồn GitHub."""))
cells.append(code_cell("""# @title Kiểm Tra và Sinh Nhãn trainIds & City Splits
import glob

%cd {ASSIGNMENT_DIR}

train_masks = glob.glob(f"{ASSIGNMENT_DIR}/dataset/**/*_gtFine_trainIds.png", recursive=True)

if len(train_masks) == 0:
    print("⚙️ Đang chạy scripts/prepare_data.py để tạo nhãn trainIds...")
    !python scripts/prepare_data.py
else:
    print(f"✅ Đã tìm thấy {len(train_masks)} masks trainIds có sẵn. Bỏ qua bước ánh xạ nhãn!")

# Tạo split theo thành phố (chống rò rỉ dữ liệu)
split_files = glob.glob(f"{ASSIGNMENT_DIR}/splits/*.txt")
if len(split_files) < 3:
    print("⚙️ Đang chạy scripts/create_splits.py...")
    !python scripts/create_splits.py
else:
    print("✅ Các file splits/ (train.txt, val.txt, test.txt) đã sẵn sàng!")"""))

# -------------------------------------------------------------
# Bước 4.1: Pre-resize đa tiến trình
# -------------------------------------------------------------
cells.append(md_cell("""## Bước 4.1: ⚡ Siêu Tốc Hóa Dữ Liệu (Fast Parallel Pre-resize 512x1024)
Thu nhỏ toàn bộ ảnh RGB và mask về 512x1024 một lần duy nhất bằng đa tiến trình CPU để loại bỏ nút thắt cổ chai nén/giải nén khi huấn luyện."""))
cells.append(code_cell("""# @title ⚡ Pre-resize Dataset Về 512x1024 Đa Tiến Trình
import os
import concurrent.futures
from pathlib import Path
from PIL import Image
from tqdm import tqdm

TARGET_W, TARGET_H = 1024, 512

def _resize_single_img(path_str):
    try:
        with Image.open(path_str) as im:
            if im.size == (TARGET_W, TARGET_H):
                return False
            im_resized = im.resize((TARGET_W, TARGET_H), Image.BILINEAR)
            im_resized.save(path_str)
        return True
    except Exception:
        return False

def _resize_single_mask(path_str):
    try:
        with Image.open(path_str) as im:
            if im.size == (TARGET_W, TARGET_H):
                return False
            im_resized = im.resize((TARGET_W, TARGET_H), Image.NEAREST)
            im_resized.save(path_str)
        return True
    except Exception:
        return False

if FAST_PRE_RESIZE:
    dataset_path = Path(ASSIGNMENT_DIR) / "dataset"
    print("🔍 Đang quét danh sách ảnh và mask cần kiểm tra kích thước...")
    all_imgs = [str(p) for p in dataset_path.rglob("*_leftImg8bit.png")]
    all_masks = [str(p) for p in dataset_path.rglob("*_gtFine_trainIds.png")]

    cpu_workers = os.cpu_count() or 4
    print(f"🚀 Đang chạy resize song song với {cpu_workers} tiến trình CPU...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_workers) as executor:
        list(tqdm(executor.map(_resize_single_img, all_imgs), total=len(all_imgs), desc="Resize RGB Images"))
        list(tqdm(executor.map(_resize_single_mask, all_masks), total=len(all_masks), desc="Resize TrainId Masks"))

    print("✅ Hoàn thành pre-resize toàn bộ dataset về 512x1024!")
else:
    print("⏩ Đã bỏ qua bước Pre-resize.")"""))

# -------------------------------------------------------------
# Bước 5: Tự động nhận diện GPU & Cập nhật configs/unet_final.yaml
# -------------------------------------------------------------
cells.append(md_cell("""## Bước 5: Nhận Diện GPU & Cập Nhật Cấu Hình `configs/unet_final.yaml`
Hỗ trợ nhận diện tự động mọi GPU, đặc biệt bao gồm dòng **GCP G4 (NVIDIA RTX PRO 6000 Blackwell 96 GB)**, **NVIDIA A100**, **NVIDIA L4**, **NVIDIA T4**."""))
cells.append(code_cell("""# @title Tự Động Tinh Chỉnh Cấu Hình unet_final.yaml Theo GPU
import os
import torch
import yaml
from pathlib import Path

cpu_cores = os.cpu_count() or 2

if not torch.cuda.is_available():
    detected_gpu_title = "CPU Fallback (Không có CUDA)"
    auto_batch_size = 4
    auto_workers = 2
    est_per_epoch = "~15 - 20 phút"
    est_total = "~10 - 12 giờ"
else:
    raw_device_name = torch.cuda.get_device_name(0).upper()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    # 1. Hỗ trợ GCP G4: NVIDIA RTX PRO 6000 Blackwell (96 GB GDDR7) hoặc RTX 6000 Ada (48 GB)
    if any(k in raw_device_name for k in ["6000", "BLACKWELL", "RTX PRO"]):
        detected_gpu_title = f"GCP G4 — NVIDIA RTX PRO 6000 ({vram_gb:.1f} GB GDDR7, Blackwell)"
        auto_batch_size = 32
        auto_workers = min(cpu_cores, 8)
        est_per_epoch = "~8 - 10 giây (Siêu tốc)" if FAST_PRE_RESIZE else "~50 - 60 giây"
        est_total = f"~5 - 7 phút ({EPOCHS} epochs)" if FAST_PRE_RESIZE else "~35 - 40 phút"

    # 2. NVIDIA A100 Tensor Core (40/80 GB SXM4)
    elif "A100" in raw_device_name:
        detected_gpu_title = f"NVIDIA A100 Tensor Core ({vram_gb:.1f} GB VRAM, HBM2e)"
        auto_batch_size = 16
        auto_workers = min(cpu_cores, 8)
        est_per_epoch = "~10 - 12 giây (Siêu tốc)" if FAST_PRE_RESIZE else "~75 - 85 giây"
        est_total = f"~7 - 9 phút ({EPOCHS} epochs)" if FAST_PRE_RESIZE else "~50 - 60 phút"

    # 3. NVIDIA L4 Tensor Core (24 GB GDDR6, Ada Lovelace)
    elif "L4" in raw_device_name:
        detected_gpu_title = f"NVIDIA L4 Tensor Core ({vram_gb:.1f} GB VRAM, Ada Lovelace)"
        auto_batch_size = 12
        auto_workers = min(cpu_cores, 6)
        est_per_epoch = "~18 - 22 giây (Siêu tốc)" if FAST_PRE_RESIZE else "~1.2 - 1.4 phút"
        est_total = f"~12 - 15 phút ({EPOCHS} epochs)" if FAST_PRE_RESIZE else "~50 - 55 phút"

    # 4. NVIDIA Tesla T4 (16 GB GDDR6, Turing)
    elif "T4" in raw_device_name:
        detected_gpu_title = f"NVIDIA Tesla T4 ({vram_gb:.1f} GB VRAM, Turing)"
        auto_batch_size = 8
        auto_workers = min(cpu_cores, 4)
        est_per_epoch = "~35 - 45 giây (Siêu tốc)" if FAST_PRE_RESIZE else "~3.5 - 4.0 phút"
        est_total = f"~25 - 30 phút ({EPOCHS} epochs)" if FAST_PRE_RESIZE else "~2.5 - 3.0 giờ"

    # 5. NVIDIA Tesla V100 (16 GB HBM2, Volta)
    elif "V100" in raw_device_name:
        detected_gpu_title = f"NVIDIA Tesla V100 ({vram_gb:.1f} GB VRAM, Volta)"
        auto_batch_size = 12
        auto_workers = min(cpu_cores, 6)
        est_per_epoch = "~15 - 20 giây"
        est_total = f"~10 - 15 phút ({EPOCHS} epochs)"

    else:
        detected_gpu_title = f"{raw_device_name} ({vram_gb:.1f} GB VRAM)"
        auto_batch_size = 32 if vram_gb >= 80 else (16 if vram_gb >= 30 else (12 if vram_gb >= 20 else (8 if vram_gb >= 12 else 4)))
        auto_workers = min(cpu_cores, 6)
        est_per_epoch = "Thích ứng"
        est_total = "Thích ứng"

if BATCH_SIZE_MODE == "Auto (Detect GPU)":
    final_batch_size = auto_batch_size
    final_workers = auto_workers
elif BATCH_SIZE_MODE == "Match Local (8)":
    final_batch_size = 8
    final_workers = auto_workers
else:
    final_batch_size = CUSTOM_BATCH_SIZE
    final_workers = CUSTOM_NUM_WORKERS

print("=" * 75)
print(f"🤖 PHẦN CỨNG ĐƯỢC NHẬN DIỆN : {detected_gpu_title}")
print(f"   Số CPU Cores khả dụng     : {cpu_cores}")
print("-" * 75)
print("⚙️  HYPERPARAMETERS HOẠT ĐỘNG:")
print(f"   • Batch Size               : {final_batch_size} ({'Tự động tối ưu' if BATCH_SIZE_MODE == 'Auto (Detect GPU)' else ('Khớp Local' if BATCH_SIZE_MODE == 'Match Local (8)' else 'Tùy chỉnh')})")
print(f"   • Số Dataloader Workers    : {final_workers}")
print(f"   • Tổng số Epochs           : {EPOCHS}")
print(f"   • Tốc độ học (AdamW LR)    : {UNET_LR}")
print(f"   • Mixed Precision (AMP)    : {USE_AMP}")
print(f"   • Random Seed              : {RANDOM_SEED}")
print("-" * 75)
print("⏱️  DỰ TÍNH THỜI GIAN HUẤN LUYỆN U-NET:")
print(f"   • Mỗi Epoch                : {est_per_epoch}")
print(f"   • Toàn bộ {EPOCHS} Epochs       : {est_total}")
print("=" * 75)

# Cập nhật trực tiếp vào file configs/unet_final.yaml
cfg_path = Path(ASSIGNMENT_DIR) / "configs" / "unet_final.yaml"
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg_data = yaml.safe_load(f)

cfg_data["batch_size"] = final_batch_size
cfg_data["num_workers"] = final_workers
cfg_data["epochs"] = EPOCHS
cfg_data["lr"] = UNET_LR
cfg_data["seed"] = RANDOM_SEED
cfg_data["amp"] = USE_AMP
cfg_data["pin_memory"] = True
cfg_data["max_train_batches"] = MAX_TRAIN_BATCHES
cfg_data["max_eval_batches"] = MAX_EVAL_BATCHES

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.dump(cfg_data, f, default_flow_style=False, sort_keys=False)

print(f"✅ Đã cập nhật cấu hình tối ưu vào file: configs/unet_final.yaml")"""))

# -------------------------------------------------------------
# Bước 6: Huấn luyện U-Net Baseline Final (gọi src/train.py)
# -------------------------------------------------------------
cells.append(md_cell("""## Bước 6: Huấn Luyện U-Net Baseline Final (Chạy Trực Tiếp `src/train.py`)
Lệnh này sẽ thực thi module `src.train` chuẩn của repo từ GitHub, sử dụng `src/models/unet.py`, `src/data.py`, `src/transforms.py`, `src/metrics.py`."""))
cells.append(code_cell("""# @title Huấn Luyện U-Net Baseline Final (Scratch)
import os

%cd {ASSIGNMENT_DIR}
print("🚀 BẮT ĐẦU HUẤN LUYỆN U-NET BASELINE BẰNG 'python -m src.train'...")
print("-" * 75)

!python -m src.train --config configs/unet_final.yaml

print("-" * 75)
print("🎉 Huấn luyện U-Net hoàn tất! Checkpoint best.pt, last.pt và log.csv đã được lưu.")"""))

# -------------------------------------------------------------
# Bước 7: Đánh giá trên tập Test bằng src/evaluate.py
# -------------------------------------------------------------
cells.append(md_cell("""## Bước 7: Đánh Giá U-Net Checkpoint Best Trên Tập Test (Chạy `src/evaluate.py`)
Thực hiện suy luận trên 500 ảnh của tập local test (`splits/test.txt`), tính toán mIoU, mDice, Pixel Accuracy và lưu kết quả vào `metrics_test.json`."""))
cells.append(code_cell("""# @title Đánh Giá U-Net Trên Tập Test
import os

%cd {ASSIGNMENT_DIR}
print("🎯 BẮT ĐẦU ĐÁNH GIÁ U-NET BẰNG 'python -m src.evaluate'...")

!python -m src.evaluate \\
  --config configs/unet_final.yaml \\
  --checkpoint checkpoints/unet_final_1024x512_s42/best.pt \\
  --split test \\
  --save_json checkpoints/unet_final_1024x512_s42/metrics_test.json

print("\\n✅ Đã xuất kết quả kiểm tra test vào checkpoints/unet_final_1024x512_s42/metrics_test.json!")"""))

# -------------------------------------------------------------
# Bước 8: Bảng so sánh đối đầu E0
# -------------------------------------------------------------
cells.append(md_cell("""## Bước 8: Bảng Tổng Hợp So Sánh Đối Đầu E0 (U-Net Baseline vs. DeepLabV3)
Tự động đối chiếu kết quả của U-Net vừa huấn luyện với các biến thể DeepLabV3 đã có sẵn kết quả test trong repo (`checkpoints/deeplabv3/`)."""))
cells.append(code_cell("""# @title Bảng Tổng Hợp Kết Quả & So Sánh Đối Đầu E0
import os
import json
import pandas as pd

models = [
    ("U-Net (Scratch Baseline)", "checkpoints/unet_final_1024x512_s42/metrics_test.json"),
    ("DeepLabV3 (Frozen Backbone)", "checkpoints/deeplabv3/local/metrics_test.json"),
    ("DeepLabV3 (Partial Fine-Tune layer 3+4)", "checkpoints/deeplabv3/resnet_layer3_4_finetune/metrics_test.json"),
    ("DeepLabV3 (Full Fine-Tune)", "checkpoints/deeplabv3/resnet50_full_deeplabv3/metrics_test.json"),
]

summary = []
for label, path in models:
    full_path = Path(ASSIGNMENT_DIR) / path
    if full_path.exists():
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary.append({
            "Mô hình (Experiment)": label,
            "Test mIoU (%)": f"{data.get('miou', data.get('eval_miou', 0)) * 100:.2f}%",
            "Mean Dice (%)": f"{data.get('mdice', data.get('eval_dice', 0)) * 100:.2f}%",
            "Pixel Acc (%)": f"{data.get('pixel_acc', 0) * 100:.2f}%",
            "Test Loss": f"{data.get('avr_loss', data.get('eval_loss', 0)):.4f}",
            "Thời gian Eval (s)": f"{data.get('eval_seconds', 0):.1f}s",
        })

if summary:
    df = pd.DataFrame(summary)
    print("=" * 80)
    print("🏆 BẢNG KẾT QUẢ SO SÁNH ĐỐI ĐẦU E0 (THIẾT KẾ THÍ NGHIỆM MỤC 5.1 HANDBOOK)")
    print("=" * 80)
    display(df)
else:
    print("Chưa tìm thấy file metrics_test.json của U-Net. Hãy chạy Bước 7 ở trên!")"""))

# -------------------------------------------------------------
# Bước 9: Đóng gói và Tải Checkpoints / Logs về máy
# -------------------------------------------------------------
cells.append(md_cell("""## Bước 9: Đóng Gói và Tải Checkpoints & Logs Về Máy Tính
Tạo file zip chứa toàn bộ kết quả `best.pt`, `last.pt`, `log.csv`, `metrics_test.json` để bạn tải về máy cá nhân hoặc lưu vào Google Drive."""))
cells.append(code_cell("""# @title Đóng Gói File Zip & Tải Về Máy Tính
import os
import zipfile
from pathlib import Path
from google.colab import files

ckpt_dir = Path(ASSIGNMENT_DIR) / "checkpoints" / "unet_final_1024x512_s42"
zip_output = "/content/assignment2_unet_checkpoints.zip"

if ckpt_dir.exists():
    all_files = [p for p in ckpt_dir.rglob("*") if p.is_file()]
    print(f"📦 Đang đóng gói {len(all_files)} files trong {ckpt_dir.name}...")
    with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in all_files:
            rel_name = p.relative_to(Path(ASSIGNMENT_DIR))
            zf.write(p, arcname=str(rel_name))
            print(f"   + {rel_name} ({p.stat().st_size / 1e6:.1f} MB)")

    zip_size_mb = os.path.getsize(zip_output) / 1e6
    print(f"\\n✅ Đã tạo thành công {zip_output} ({zip_size_mb:.1f} MB)")
    print("🚀 Đang kích hoạt tải về máy tính qua trình duyệt...")
    files.download(zip_output)
else:
    print(f"❌ Không tìm thấy thư mục: {ckpt_dir}")"""))

# Save notebook
output_nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": []
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

nb_out_path = Path("colab_U_net.ipynb")
with open(nb_out_path, "w", encoding="utf-8") as f:
    json.dump(output_nb, f, indent=1, ensure_ascii=False)

print("Created completely modular and Git-synced colab_U_net.ipynb successfully!")
