package com.example.mobilesam

import android.graphics.Bitmap

/**
 * Thin orchestrator over a pluggable [Segmenter]: runs segmentation and composes
 * the overlay. The heavy per-frame logic lives in the segmenter; this class only
 * maps [SegmentationOutput] to a [Result] and serializes access so ONNX sessions
 * are never used from two threads at once.
 */
class InferencePipeline(
    private var segmenter: Segmenter,
) {

    data class Result(
        val overlay: Bitmap,
        val objectCount: Int,
        val detectedLabels: List<String>,
        val detectedClasses: List<Int>,
        val yoloMs: Long,
        val encMs: Long,
        val decMs: Long,
        val compMs: Long,
        val totalMs: Long,
        val encodedThisFrame: Boolean,
        val embeddingReady: Boolean,
    )

    /** Camera path: streaming segmentation + compose. */
    @Synchronized
    fun onFrame(source: Bitmap): Result {
        val start = System.currentTimeMillis()
        val out = segmenter.segmentStreaming(source)
        val tComp0 = System.currentTimeMillis()
        val overlay = MaskComposer.composeOpaque(source, out.masks, out.boxes)
        val compMs = System.currentTimeMillis() - tComp0
        val elapsed = System.currentTimeMillis() - start

        android.util.Log.d(
            "MobileSAM",
            "boxes=${out.boxes.size} ready=${out.embeddingReady} yolo=${out.yoloMs}ms " +
                "enc=${out.encMs}ms dec=${out.decMs}ms comp=${compMs}ms total=${elapsed}ms")
        return Result(
            overlay = overlay,
            objectCount = out.masks.size,
            detectedLabels = out.boxes.map { CocoLabels.labelFor(it.classId) }.distinct(),
            detectedClasses = out.boxes.map { it.classId }.distinct(),
            yoloMs = out.yoloMs,
            encMs = out.encMs,
            decMs = out.decMs,
            compMs = compMs,
            totalMs = elapsed,
            encodedThisFrame = out.encodedThisFrame,
            embeddingReady = out.embeddingReady,
        )
    }

    /** Photo path: single full segmentation + compose. */
    @Synchronized
    fun runSingle(source: Bitmap): Result {
        val start = System.currentTimeMillis()
        val out = segmenter.segmentFull(source)
        val tComp0 = System.currentTimeMillis()
        val overlay = MaskComposer.composeOpaque(source, out.masks, out.boxes)
        val compMs = System.currentTimeMillis() - tComp0
        val elapsed = System.currentTimeMillis() - start

        android.util.Log.d(
            "MobileSAM",
            "single boxes=${out.boxes.size} yolo=${out.yoloMs}ms enc=${out.encMs}ms " +
                "dec=${out.decMs}ms comp=${compMs}ms total=${elapsed}ms")
        return Result(
            overlay = overlay,
            objectCount = out.masks.size,
            detectedLabels = out.boxes.map { CocoLabels.labelFor(it.classId) }.distinct(),
            detectedClasses = out.boxes.map { it.classId }.distinct(),
            yoloMs = out.yoloMs,
            encMs = out.encMs,
            decMs = out.decMs,
            compMs = compMs,
            totalMs = elapsed,
            encodedThisFrame = out.encodedThisFrame,
            embeddingReady = out.embeddingReady,
        )
    }

    /**
     * Swap the active segmenter. Runs on the same lock as [onFrame]/[runSingle]
     * so the old segmenter is never mid-run when closed. The old segmenter's
     * sessions are released inside the lock; no explicit GC here — it would
     * stall the analyzer thread on the hot path.
     */
    @Synchronized
    fun switchSegmenter(newSegmenter: Segmenter) {
        val old = segmenter
        segmenter = newSegmenter
        old.close()
    }

    @Synchronized
    fun close() {
        segmenter.close()
    }
}
