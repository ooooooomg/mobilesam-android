package com.example.mobilesam

import android.graphics.Bitmap

/**
 * The two-stage MobileSAM pipeline: YOLO detection -> image encoder -> mask
 * decoder per box.
 *
 * Performance strategy (moved here from the old InferencePipeline): the encoder
 * is the dominant cost, so it re-runs only every ENCODE_INTERVAL frames and the
 * cached embedding is reused between encodes while YOLO keeps tracking every
 * frame.
 */
class TwoStageSegmenter(
    private val yolo: YoloDetector,
    private val encoder: SamImageEncoder,
    private val decoder: SamMaskDecoder,
) : Segmenter {

    companion object {
        private const val ENCODE_INTERVAL = 6
    }

    private var cachedEmbedding: FloatArray? = null
    private var cachedPrep: SamImageEncoder.Preprocessed? = null
    private var frameCounter = 0

    override fun segmentStreaming(source: Bitmap): SegmentationOutput {
        val start = System.currentTimeMillis()
        val encodeThis = frameCounter % ENCODE_INTERVAL == 0
        frameCounter++

        val boxes = yolo.detect(source)
        val tYolo = System.currentTimeMillis() - start

        var prep = cachedPrep
        if (encodeThis || cachedEmbedding == null) {
            val (emb, p) = encoder.encode(source)
            cachedEmbedding = emb
            cachedPrep = p
            prep = p
        }
        val embedding = cachedEmbedding
        val tEnc = System.currentTimeMillis() - start - tYolo
        val ready = embedding != null

        val tDec0 = System.currentTimeMillis()
        val masks = ArrayList<SamMaskDecoder.SegmentedMask>(boxes.size)
        val srcW = source.width
        val srcH = source.height
        if (ready) {
            for (box in boxes) {
                if (box.x2 - box.x1 < 5f || box.y2 - box.y1 < 5f) continue
                masks.add(decoder.decode(embedding!!, box, prep!!.scale, srcW, srcH))
            }
        }
        val tDec = System.currentTimeMillis() - tDec0

        return SegmentationOutput(
            masks = masks,
            boxes = boxes,
            yoloMs = tYolo,
            encMs = tEnc,
            decMs = tDec,
            encodedThisFrame = encodeThis,
            embeddingReady = ready,
        )
    }

    override fun segmentFull(source: Bitmap): SegmentationOutput {
        val start = System.currentTimeMillis()

        val boxes = yolo.detect(source)
        val tYolo = System.currentTimeMillis() - start

        val tEnc0 = System.currentTimeMillis()
        val (embedding, prep) = encoder.encode(source)
        val tEnc = System.currentTimeMillis() - tEnc0

        val tDec0 = System.currentTimeMillis()
        val masks = ArrayList<SamMaskDecoder.SegmentedMask>(boxes.size)
        for (box in boxes) {
            if (box.x2 - box.x1 < 5f || box.y2 - box.y1 < 5f) continue
            masks.add(decoder.decode(embedding, box, prep.scale, source.width, source.height))
        }
        val tDec = System.currentTimeMillis() - tDec0

        return SegmentationOutput(
            masks = masks,
            boxes = boxes,
            yoloMs = tYolo,
            encMs = tEnc,
            decMs = tDec,
            encodedThisFrame = true,
            embeddingReady = true,
        )
    }

    override fun close() {
        yolo.close()
        encoder.close()
        decoder.close()
    }
}
