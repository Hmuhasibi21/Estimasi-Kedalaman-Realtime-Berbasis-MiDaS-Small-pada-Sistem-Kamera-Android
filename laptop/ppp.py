import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------
# Tahap 1: Inisialisasi Model MiDaS
# -------------------------------------------------------------------
print("\n=== Inisialisasi Model MiDaS ===")
print("Pilih model:")
print("1. MiDaS_small (lebih cepat, lebih ringan)")
print("2. DPT_Large (lebih akurat, tapi lebih berat)")
choice = input("Masukkan pilihan (1/2): ").strip()

if choice == "2":
    model_type = "DPT_Large"
else:
    model_type = "MiDaS_small"

print(f"\nMemuat model {model_type}...\n")

try:
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # -------------------------------------------------------------------
    # Tahap 1.5: Deteksi dan Pilih Device Otomatis (GPU/CPU)
    # -------------------------------------------------------------------
    print("\n=========🔍 Mengecek perangkat yang tersedia...=========\n")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n=========✅ Menggunakan GPU NVIDIA: {torch.cuda.get_device_name(0)}=========\n")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n=========✅ Menggunakan GPU Apple (MPS)=========\n")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"\n=========✅ Menggunakan GPU Intel (XPU)=========\n")
    else:
        device = torch.device("cpu")
        print(f"\n⚙️  GPU tidak tersedia, menggunakan CPU\n")

    # Kirim model ke device
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "MiDaS_small":
        transform = midas_transforms.small_transform
    else:
        transform = midas_transforms.dpt_transform

    print(f"✅ Model '{model_type}' berhasil dimuat di {device}\n")

except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    exit()

# -------------------------------------------------------------------
# Tahap 2: Membuka Kamera
# -------------------------------------------------------------------
print("🎥 Membuka kamera...\n")
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("https://10.200.234.249:8080/video")
if not cap.isOpened():
    print("❌ Tidak dapat membuka kamera.")
    exit()

# -------------------------------------------------------------------
# Tahap 3: Setup Matplotlib untuk UI
# -------------------------------------------------------------------
plt.ion()
fig, (ax_frame, ax_hist, ax_flow) = plt.subplots(1, 3, figsize=(15, 5))
flow_text = "Kamera → Preprocessing → Model MiDaS → Depth Map → Visualisasi"

# -------------------------------------------------------------------
# Tahap 4: Loop Utama (Realtime Processing)
# -------------------------------------------------------------------
frame_count = 0
start_time = time.time()

print("🟢 Tekan [S] untuk simpan frame, [Q] untuk keluar.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame dari kamera.")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_normalized = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # Gabung kamera dan depth map
    combined_display = np.hstack((frame, depth_colored))
    max_width = 1280
    scale = max_width / combined_display.shape[1] if combined_display.shape[1] > max_width else 1.0
    if scale < 1.0:
        new_dim = (int(combined_display.shape[1]*scale), int(combined_display.shape[0]*scale))
        combined_display = cv2.resize(combined_display, new_dim, interpolation=cv2.INTER_AREA)

    # OpenCV Window
    cv2.imshow(f"Kamera (Kiri) | Depth Map {model_type} (Kanan)", combined_display)

    # ----------------------------------------------------------------
    # Matplotlib UI (Realtime)
    # ----------------------------------------------------------------
    ax_frame.clear()
    ax_frame.imshow(depth_colored[..., ::-1])
    ax_frame.set_title(f"Depth Map ({model_type})")

    ax_hist.clear()
    ax_hist.hist(depth_map.ravel(), bins=60, color="blue", alpha=0.7)
    ax_hist.set_title("Distribusi Kedalaman")
    ax_hist.set_xlabel("Nilai Depth (pixel)")
    ax_hist.set_ylabel("Frekuensi")

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    ax_flow.clear()
    ax_flow.axis("off")
    ax_flow.text(0.5, 0.65, flow_text, ha="center", va="center", fontsize=11, wrap=True)
    ax_flow.text(0.5, 0.3,
                 f"Device: {device}\nModel: {model_type}\nFPS: {fps:.2f}\nResolusi: {frame.shape[1]}x{frame.shape[0]}",
                 ha="center", va="center", fontsize=10)

    plt.pause(0.01)

    # ----------------------------------------------------------------
    # KONTROL INPUT
    # ----------------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n🔴 Keluar dari program...")
        break
    elif key == ord('s'):
        filename = f"screenshot_{model_type}.png"
        cv2.imwrite(filename, combined_display)
        np.save(f"depth_{model_type}.npy", depth_map)
        print(f"💾 Screenshot & data depth disimpan sebagai {filename}")

# -------------------------------------------------------------------
# Tahap 5: Membersihkan
# -------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
plt.close(fig)
print("✅ Program selesai.")
