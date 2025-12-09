import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
#  Tahap 1: Inisialisasi Model MiDaS
# -------------------------------------------------------------------
print("\nSedang memuat model MiDaS...\n")
try:
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform
    print(f"\nModel berhasil dimuat dan berjalan di {device}.\n")

except Exception as e:
    print(f"Gagal memuat model: {e}")
    exit()

# -------------------------------------------------------------------
#  Tahap 2: Membuka Kamera
# -------------------------------------------------------------------
print("\nMembuka kamera...\n")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

# -------------------------------------------------------------------
#  Setup Matplotlib untuk grafik
# -------------------------------------------------------------------
plt.ion()  # interactive mode on
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')  # garis grafik
ax.set_xlim(0, 255)  # range nilai depth (normalisasi 0–255)
ax.set_ylim(0, 10000)  # estimasi jumlah piksel (sesuaikan nanti)
ax.set_xlabel("Nilai Kedalaman (0=dekat, 255=jauh)")
ax.set_ylabel("Jumlah Piksel")
ax.set_title("Distribusi Depth Map")

# -------------------------------------------------------------------
#  Tahap 3: Loop Utama
# -------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        break

    frame = cv2.flip(frame, 1)

    # --- PREPROCESSING ---
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    # --- INFERENCE ---
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # --- POSTPROCESSING ---
    output_display = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    output_display = cv2.applyColorMap(output_display, cv2.COLORMAP_JET)

    # --- VISUALISASI HASIL ---
    combined_display = np.hstack((frame, output_display))
    cv2.imshow('Kamera Asli (Kiri) | Depth Map MiDaS (Kanan)', combined_display)

    # --- UPDATE GRAFIK DEPTH ---
    hist, bins = np.histogram(output_display.ravel(), bins=256, range=[0, 256])
    line.set_xdata(bins[:-1])
    line.set_ydata(hist)
    ax.set_ylim(0, hist.max() + 1000)  # update skala Y sesuai data
    fig.canvas.draw()
    fig.canvas.flush_events()

    # --- KONTROL INPUT ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("screenshot_frame.png", combined_display)
        print("Screenshot disimpan sebagai screenshot_frame.png")

# -------------------------------------------------------------------
#  Tahap 4: Membersihkan
# -------------------------------------------------------------------
print("\nMenutup program...\n")
cap.release()
cv2.destroyAllWindows()
plt.close(fig)
