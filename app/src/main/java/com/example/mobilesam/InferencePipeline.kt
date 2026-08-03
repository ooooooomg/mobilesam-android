package com.example.mobilesam

import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap

/**
 * Orchestrates YOLO -> MobileSAM encoder -> MobileSAM decoder -> overlay.
 *
 * Performance strategy: the MobileSAM encoder is the dominant cost
 * (~600-800ms on phone CPU at 512px). To sustain interactive framerates we
 * re-encode only every ENCODE_INTERVAL frames; between encodes we reuse the
 * cached embedding while YOLO keeps tracking object boxes every frame. This
 * trades a small temporal lag on the mask for a large frame-rate win.
 */
class InferencePipeline(
    private val encoder: SamImageEncoder,
    private val decoder: SamMaskDecoder,
    private val yolo: YoloDetector,
) {

    data class Result(
        val overlay: Bitmap,
        val objectCount: Int,
        val inferenceMs: Long,
        val encodedThisFrame: Boolean,
    )

    companion object {
        // Re-encode the image every N frames (tune for device).
        private const val ENCODE_INTERVAL = 6
    }

    private var cachedEmbedding: FloatArray? = null
    private var cachedPrep: SamImageEncoder.Preprocessed? = null
    private var frameCounter = 0

    /**
     * Run the full pipeline on one bitmap (single-threaded).
     *
     * @param source RGB bitmap. Output overlay shares its resolution.
     */
    @Synchronized
    fun run(source: Bitmap): Result {
        val start = System.currentTimeMillis()
        val encodeThis = frameCounter % ENCODE_INTERVAL == 0
        frameCounter++

        // 1. Detect objects every frame (cheap: YOLO 320).
        val boxes = yolo.detect(source)
        val tYolo = System.currentTimeMillis() - start

        // 2. Encode only every ENCODE_INTERVAL-th frame; reuse cached
        //    embedding otherwise.
        var prep = cachedPrep
        if (encodeThis || cachedEmbedding == null) {
            val (emb, p) = encoder.encode(source)
            cachedEmbedding = emb
            cachedPrep = p
            prep = p
        }
        val embedding = cachedEmbedding ?: return Result(
            Bitmap.createBitmap(source), 0, System.currentTimeMillis() - start, false)
        val tEnc = System.currentTimeMillis() - start - tYolo

        // 3. Decode a mask per box.
        val tDec0 = System.currentTimeMillis()
        val masks = ArrayList<SamMaskDecoder.SegmentedMask>(boxes.size)
        val srcW = source.width
        val srcH = source.height
        for (box in boxes) {
            if (box.x2 - box.x1 < 5f || box.y2 - box.y1 < 5f) continue
            val seg = decoder.decode(
                embedding, box, prep!!.scale, srcW, srcH
            )
            masks.add(seg)
        }
        val tDec = System.currentTimeMillis() - tDec0

        // 4. Compose overlay.
        val tComp0 = System.currentTimeMillis()
        val overlay = MaskComposer.compose(source, masks, boxes)
        val tComp = System.currentTimeMillis() - tComp0

        val elapsed = System.currentTimeMillis() - start
        android.util.Log.d(
            "MobileSAM",
            "encode=$encodeThis boxes=${boxes.size} yolo=${tYolo}ms enc=${tEnc}ms " +
                "dec=${tDec}ms comp=${tComp}ms total=${elapsed}ms")
        return Result(overlay, masks.size, elapsed, encodeThis)
    }

    fun close() {
        yolo.close()
        encoder.close()
        decoder.close()
    }
}
