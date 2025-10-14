//MyLibrary/src/main/java/com/davoice/speakerid/VadAdapter.java
package ai.perplexity.hotword.speakerid;

import android.content.Context;
import ai.onnxruntime.OrtException;
import java.io.Closeable;

/** Adapter that makes VadDetector look like the simple Vad interface. */
public final class VadAdapter implements Vad, Closeable {
    private final VadDetector det;
    public final int frameLength;   // samples per VAD call (Constants.FRAME_LENGTH)
    public final int sampleRate;    // Hz (Constants.SAMPLE_RATE)

    public VadAdapter(Context ctx, String modelPath) throws OrtException {
        this.det = new VadDetector(ctx, modelPath);
        this.frameLength = Constants.FRAME_LENGTH;
        this.sampleRate  = Constants.SAMPLE_RATE;
    }

    /** Optional, if you want to explicitly clear the model’s internal state. */
    public void reset() { det.reset(); }

    /** Feed one block of PCM16 at sampleRate; returns speech probability [0..1]. */
    @Override public float feed(short[] block16k) {
        // Your detector can handle mismatched lengths (it reallocates),
        // but we strongly recommend using exactly frameLength each call.
        return det.predict_2(block16k);
    }

    @Override public void close() {
        try { det.close(); } catch (Exception ignore) {}
    }
}

