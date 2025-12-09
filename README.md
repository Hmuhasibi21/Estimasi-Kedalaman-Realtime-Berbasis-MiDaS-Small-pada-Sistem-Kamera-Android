# Real-Time Monocular Depth Estimation on Android (MiDaS-Small)

![Android](https://img.shields.io/badge/Platform-Android-3DDC84?style=flat&logo=android)
![Kotlin](https://img.shields.io/badge/Language-Kotlin-7F52FF?style=flat&logo=kotlin)
![TensorFlow Lite](https://img.shields.io/badge/ML-TensorFlow%20Lite-FF6F00?style=flat&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?style=flat&logo=opencv)

This repository contains an Android application implementation for **Real-Time Monocular Depth Estimation** using the **MiDaS-Small v2.1** model. The system is designed to run efficiently on mobile devices with limited computational resources, providing a side-by-side view of the live camera feed and the estimated depth map.

This project was developed by **Kelompok 3** from **Universitas Brawijaya** as a final implementation report.

## 📱 Project Overview

Depth estimation is a crucial task in computer vision, enabling machines to understand the 3D structure of a scene. This project implements the **MiDaS (Mixed Depth Scale)** model—specifically the lightweight **MiDaS-Small** variant—to achieve zero-shot depth estimation on Android smartphones without requiring dedicated depth sensors (like LiDAR).

### Key Features
* **Real-Time Inference:** Processes live camera frames using the MiDaS-Small model converted to TFLite.
* **Side-by-Side Visualization:** Displays the original RGB frame and the generated Depth Map simultaneously.
* **Depth Visualization:** Uses the **JET Colormap** to represent depth (Red = Near, Blue = Far).
* **Performance Metrics:** Real-time display of **FPS** and **Latency** (ms) on the UI.
* **Optimized Pipeline:** Implements frame skipping and efficient bitmap handling to manage thermal and battery constraints.

## 📸 Screenshots & Demo

| Depth Map Output |
|:---:|
| ![Depth Map](docs/screenshot_depth.jpg) |

*(Note: The depth map visualizes proximity using a heatmap: **Red/Orange** indicates close objects, while **Blue** indicates the background)*.

## 🛠️ Tech Stack & Architecture

This project is built using the following technologies:

* **Language:** Kotlin
* **IDE:** Android Studio
* **Camera:** Android CameraX API (ImageAnalysis & Preview)
* **ML Inference:** TensorFlow Lite (Model converted from PyTorch)
* **Image Processing:** OpenCV for Android (Post-processing & Color mapping)

### System Pipeline
1.  **Input:** Camera capture via CameraX (Target resolution: 320x240).
2.  **Preprocessing:** Resize (256x256) & Normalize Tensor.
3.  **Inference:** MiDaS-Small execution via TFLite Interpreter.
4.  **Post-Processing:** Output normalization and OpenCV Colormap application (JET).
5.  **Display:** Render resulting Bitmap to UI.

## 📊 Performance Benchmarks

We evaluated the application holistically (End-to-End) on a **Samsung Note 20 Ultra** (GPU: Adreno 650).

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Average Latency** | **435 ms** | Time taken for preprocessing, inference, and post-processing per frame. |
| **Throughput** | **1.60 FPS** | Real-world frames per second displayed to the user. |

*Note: The performance reflects the heavy computational load of running a Transformer-based model (even the Small variant) on a mobile CPU/GPU.*

## 🚀 Getting Started

### Prerequisites
* Android Studio Iguana or newer.
* Android Device with Camera support (Min SDK 24 recommended).
* Internet connection (to sync Gradle dependencies).

### Installation
1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/monocular-depth-estimation-android.git](https://github.com/your-username/monocular-depth-estimation-android.git)
    ```
2.  **Open in Android Studio**
    * Open Android Studio -> File -> Open -> Select the cloned folder.
3.  **Sync Gradle**
    * Wait for the project to download dependencies (CameraX, TensorFlow Lite, OpenCV).
4.  **Add the Model**
    * Ensure the `midas_small.tflite` file is located in `app/src/main/assets/`.
5.  **Run**
    * Connect your Android device via USB (make sure USB Debugging is ON).
    * Click the **Run** (Play) button.

## 📂 Project Structure

---

## 👥 Authors (Kelompok 3)

* **Abdul Haris Muhasibi** - 235150307111047
* **Muhfi Fawwaz Rizqullah** - 235150307111009

**Program Studi Teknik Komputer**
**Fakultas Ilmu Komputer - Universitas Brawijaya**
**2025**

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---
*Based on the Final Report: "Implementasi Estimasi Kedalaman Realtime Berbasis Model MiDaS Pada Sistem Kamera Menggunakan Arsitektur MiDaS Small"*
