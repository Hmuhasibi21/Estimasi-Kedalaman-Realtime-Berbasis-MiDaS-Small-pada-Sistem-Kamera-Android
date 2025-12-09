package com.example.midas_small

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    // Komponen UI
    private lateinit var viewFinder: PreviewView
    private lateinit var imageView: ImageView
    private lateinit var fpsTextView: TextView
    private lateinit var latencyTextView: TextView // BARU: Teks latensi
    private lateinit var switchCameraButton: ImageButton
    private lateinit var switchColormapButton: ImageButton

    // Executor untuk background thread
    private lateinit var cameraExecutor: ExecutorService
    private var module: Module? = null

    // Konstanta Izin
    private val REQUEST_CODE_PERMISSIONS = 10
    private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

    // --- PERUBAHAN UNTUK OPTIMASI ---
    private var frameCounter = 0
    private val FRAME_SKIPPING_RATE = 10

    // --- VARIABEL BARU UNTUK FPS ---
    private var processedFrameCounter = 0
    private var lastFpsTimestamp: Long = 0

    // --- BARU: Variabel state untuk fitur baru ---
    private var cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
    private var colormapIndex = 0
    private val colormapList = listOf(
        Pair("JET", Imgproc.COLORMAP_JET),
        Pair("INFERNO", Imgproc.COLORMAP_INFERNO),
        Pair("BONE", Imgproc.COLORMAP_BONE),
        Pair("WINTER", Imgproc.COLORMAP_WINTER),
        Pair("GRAYSCALE", -1)
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Inisialisasi OpenCV
        if (OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "OpenCV initialized successfully")
        } else {
            Log.e("OpenCV", "OpenCV initialization failed")
        }

        viewFinder = findViewById(R.id.viewFinder)
        imageView = findViewById(R.id.imageView)
        fpsTextView = findViewById(R.id.fpsTextView)
        latencyTextView = findViewById(R.id.latencyTextView) // BARU: Inisialisasi latensi

        // Inisialisasi tombol
        switchCameraButton = findViewById(R.id.switchCameraButton)
        switchColormapButton = findViewById(R.id.switchColormapButton)

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        lastFpsTimestamp = System.currentTimeMillis()

        // Set listener untuk tombol
        switchCameraButton.setOnClickListener { switchCamera() }
        switchColormapButton.setOnClickListener { switchColormap() }

        Thread {
            try {
                module = loadModel("midas_mobile.ptl")
                Log.d("Pytorch", "Model loaded successfully")
            } catch (e: Exception) {
                Log.e("Pytorch", "Error loading model!", e)
            }
        }.start()
    }

    // Fungsi untuk ganti kamera
    private fun switchCamera() {
        cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }
        startCamera()
    }

    // Fungsi untuk ganti colormap
    private fun switchColormap() {
        colormapIndex++
        if (colormapIndex >= colormapList.size) {
            colormapIndex = 0
        }
        val (name, _) = colormapList[colormapIndex]
        Toast.makeText(this, "Colormap: $name", Toast.LENGTH_SHORT).show()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetResolution(Size(320, 240))
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->

                        frameCounter++
                        if (frameCounter % FRAME_SKIPPING_RATE != 0) {
                            image.close()
                            return@setAnalyzer
                        }
                        if (module == null) {
                            image.close()
                            return@setAnalyzer
                        }

                        val bitmap = image.toBitmap()
                        if (bitmap != null) {
                            // BARU: Terima Pair (Bitmap dan Long) dari runInference
                            val (resultBitmap, latencyInMs) = runInference(bitmap)

                            // BARU: Hanya lanjut jika inferensi berhasil
                            if (resultBitmap != null) {
                                // --- (Logika penghitungan FPS tetap sama) ---
                                processedFrameCounter++
                                val currentTime = System.currentTimeMillis()
                                val elapsedTime = currentTime - lastFpsTimestamp
                                var fpsTextToDisplay: String? = null

                                // BARU: Siapkan teks latensi
                                val latencyTextToDisplay = "Latency: ${latencyInMs}ms"

                                if (elapsedTime >= 1000) {
                                    val fps = processedFrameCounter / (elapsedTime / 1000.0)
                                    fpsTextToDisplay = "FPS: %.2f".format(fps)
                                    processedFrameCounter = 0
                                    lastFpsTimestamp = currentTime
                                }

                                runOnUiThread {
                                    imageView.setImageBitmap(resultBitmap)
                                    latencyTextView.text = latencyTextToDisplay // BARU: Tampilkan latensi

                                    if (fpsTextToDisplay != null) {
                                        fpsTextView.text = fpsTextToDisplay
                                    }
                                }
                            }
                        }
                        image.close()
                    }
                }

            val currentCameraSelector = cameraSelector
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, currentCameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e("CameraX", "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // --- FUNGSI DIPERBARUI (runInference) UNTUK MENGEMBALIKAN LATENCY ---
    private fun runInference(bitmap: Bitmap): Pair<Bitmap?, Long> {
        val startTime = System.nanoTime() // BARU: Mulai timer (lebih presisi)

        var mat: Mat? = null
        var normalizedMat: Mat? = null
        var intMat: Mat? = null
        var coloredMat: Mat? = null
        var finalMatForBitmap: Mat? = null

        try {
            // PREPROCESSING
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true)
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedBitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )

            // INFERENCE
            val outputTensor = module?.forward(IValue.from(inputTensor))?.toTensor()
            val outputArray = outputTensor?.dataAsFloatArray ?: return Pair(null, 0) // Gagal

            // POST-PROCESSING DENGAN OPENCV UNTUK WARNA
            val width = outputTensor.shape()[2].toInt()
            val height = outputTensor.shape()[1].toInt()

            mat = Mat(height, width, CvType.CV_32FC1)
            mat.put(0, 0, outputArray)

            normalizedMat = Mat()
            Core.normalize(mat, normalizedMat, 0.0, 255.0, Core.NORM_MINMAX)

            intMat = Mat()
            normalizedMat.convertTo(intMat, CvType.CV_8UC1)

            val selectedColormapCode = colormapList[colormapIndex].second
            finalMatForBitmap = Mat()

            if (selectedColormapCode == -1) {
                Imgproc.cvtColor(intMat, finalMatForBitmap, Imgproc.COLOR_GRAY2RGBA)
            } else {
                coloredMat = Mat()
                Imgproc.applyColorMap(intMat, coloredMat, selectedColormapCode)
                Imgproc.cvtColor(coloredMat, finalMatForBitmap, Imgproc.COLOR_BGR2RGBA)
            }

            val resultBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(finalMatForBitmap, resultBitmap)

            val endTime = System.nanoTime() // BARU: Akhiri timer
            val latencyInMs = (endTime - startTime) / 1_000_000 // BARU: Hitung latensi dalam milidetik

            return Pair(resultBitmap, latencyInMs) // BARU: Kembalikan hasil dan latensi

        } catch (e: Exception) {
            Log.e("Inference", "Error during inference", e)
            return Pair(null, 0) // Gagal
        } finally {
            mat?.release()
            normalizedMat?.release()
            intMat?.release()
            coloredMat?.release()
            finalMatForBitmap?.release()
        }
    }

    // --- (Fungsi helper lainnya tetap sama) ---

    @Throws(Exception::class)
    private fun loadModel(assetName: String): Module {
        val file = File(filesDir, assetName)
        if (!file.exists()) {
            assets.open(assetName).use { `is` ->
                FileOutputStream(file).use { os ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (`is`.read(buffer).also { read = it } != -1) {
                        os.write(buffer, 0, read)
                    }
                }
            }
        }
        return Module.load(file.absolutePath)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}