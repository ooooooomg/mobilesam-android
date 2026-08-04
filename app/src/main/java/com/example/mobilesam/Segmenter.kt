package com.example.mobilesam

import android.graphics.Bitmap

/**
 * Unified segmentation pipeline output. Both the two-stage MobileSAM pipeline
 * and end-to-end YOLO-seg models produce this shape, so the composer and the
 * activities don't care which model is active.
 */
data class SegmentationOutput(
    val masks: List<SamMaskDecoder.SegmentedMask>,
    val boxes: List<YoloDetector.DetBox>,
    val yoloMs: Long,
    val encMs: Long,
    val decMs: Long,
    val encodedThisFrame: Boolean,
    val embeddingReady: Boolean,
)

/**
 * A pluggable segmentation model. Implementations own their ONNX sessions and
 * any frame-to-frame caches (e.g. MobileSAM's re-encode-every-N-frames cache).
 */
interface Segmenter {
    /** Camera path: may reuse cached state between frames. */
    fun segmentStreaming(source: Bitmap): SegmentationOutput

    /** Photo path: always runs the full model (no cache reuse). */
    fun segmentFull(source: Bitmap): SegmentationOutput

    /** Release this segmenter's ONNX sessions. Must be called while no other
     *  thread is running the segmenter (guaranteed by the pipeline lock). */
    fun close()
}
