package ai.perplexity.hotword.speakerid;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

import androidx.annotation.RequiresPermission;
import androidx.core.content.ContextCompat;

public final class MicRecorder implements AutoCloseable {
    private static final String TAG = "MicRecorder";

    private final Context appContext;
    private final int sampleRate;
    private final int blockSamples;
    private AudioRecord rec;

    public MicRecorder(Context context, int sampleRate, int blockSamples) {
        this.appContext = context.getApplicationContext();
        this.sampleRate = sampleRate;
        this.blockSamples = blockSamples;
    }

    /**
     * Starts recording if RECORD_AUDIO permission is granted.
     * Throws SecurityException with a descriptive message if not granted or creation fails.
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    public void start() {
        // 1) Check permission explicitly (satisfies Lint + avoids runtime crash)
        if (ContextCompat.checkSelfPermission(appContext, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            throw new SecurityException("RECORD_AUDIO permission not granted. " +
                    "Caller must request it before starting MicRecorder.");
        }

        // 2) Compute buffer size
        int min = AudioRecord.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT);

        if (min <= 0) {
            // defensive fallback
            min = blockSamples * 2;
        }
        int buf = Math.max(min, blockSamples * 4);

        // 3) Create AudioRecord (catch SecurityException to provide a clearer message)
        try {
            rec = new AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    buf
            );
        } catch (SecurityException se) {
            throw new SecurityException("Failed to create AudioRecord: " + se.getMessage(), se);
        }

        // 4) Validate state
        if (rec.getState() != AudioRecord.STATE_INITIALIZED) {
            // Some devices reject VOICE_RECOGNITION; try MIC as a fallback
            Log.w(TAG, "VOICE_RECOGNITION source not initialized; retrying with MIC");
            try {
                if (rec != null) { rec.release(); rec = null; }
                rec = new AudioRecord(
                        MediaRecorder.AudioSource.MIC,
                        sampleRate,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        buf
                );
            } catch (SecurityException se) {
                throw new SecurityException("Failed to create AudioRecord (MIC): " + se.getMessage(), se);
            }
            if (rec.getState() != AudioRecord.STATE_INITIALIZED) {
                if (rec != null) { rec.release(); rec = null; }
                throw new IllegalStateException("AudioRecord not initialized with any source.");
            }
        }

        rec.startRecording();
    }

    /** Reads exactly one block; may busy-wait slightly to fill it. */
    public short[] readBlock() {
        if (rec == null) throw new IllegalStateException("MicRecorder not started.");
        short[] b = new short[blockSamples];
        int off = 0;
        while (off < b.length) {
            int n = rec.read(b, off, b.length - off);
            if (n > 0) off += n;
        }
        return b;
    }

    public void stop() {
        if (rec != null) {
            try {
                rec.stop();
            } catch (Exception ignore) {}
            try {
                rec.release();
            } catch (Exception ignore) {}
            rec = null;
        }
    }

    @Override public void close() { stop(); }
}
