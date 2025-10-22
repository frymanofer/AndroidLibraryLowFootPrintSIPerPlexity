package ai.perplexity.hotword.speakerid;

import ai.onnxruntime.*;
import android.util.Log;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.*;

/**
 * Minimal ORT wrapper for NeMo Speakernet ONNX on Android.
 * Handles waveform or fbank features. For models logging "audio_signal [-1, 64, -1]"
 * we feed [1, 64, T] (mel-major) and optionally provide `length=[T]`.
 */
public final class SpeakerEmbedderOrt implements AutoCloseable {
    private static final String TAG = "SpeakerEmbedderOrt";

    private final OrtEnvironment env;
    private final OrtSession session;
    private final int sampleRate;

    // Resolved inputs
    private final String dataInputName;
    private final long[] dataInputShape;
    private final boolean expectsFeatures;
    private final String lengthInputName;  // nullable
    private final boolean lengthIsInt64;

    // Feature layout detection
    // true  -> [B, 64, T]
    // false -> [B, T, 64]
    private final boolean melAtDim1;

    // FBANK generator (produces 80 mels); we slice to 64 if model wants 64.
    private final Fbank fbank;
    private static final int SRC_MELS = 80;
    private static final int DST_MELS = 64;

    // Cache output names once (constructor can throw OrtException)
    private final List<String> outputNames;

    // Streaming buffer (PCM float32 in [-1,1])
    private final ArrayList<Float> pcm = new ArrayList<>();
    private boolean finished = false;

    public SpeakerEmbedderOrt(OrtEnvironment env,
                              String speakernetOnnxPath,
                              OrtSession.SessionOptions opts,
                              int sampleRateHz) throws OrtException {
        this.env = env;
        this.sampleRate = sampleRateHz <= 0 ? 16000 : sampleRateHz;
        this.session = env.createSession(speakernetOnnxPath, opts);

        // Detect inputs
        String foundData = null;
        long[] foundShape = null;
        boolean foundFeatures = false;
        String foundLen = null;
        boolean lenIsI64 = true;
        boolean tmpMelAtDim1 = false;

        Map<String, NodeInfo> inInfo = session.getInputInfo();
        for (Map.Entry<String, NodeInfo> e : inInfo.entrySet()) {
            String name = e.getKey();
            NodeInfo ni = e.getValue();
            if (!(ni.getInfo() instanceof TensorInfo)) continue;
            TensorInfo ti = (TensorInfo) ni.getInfo();
            long[] sh = ti.getShape();
            String lname = name.toLowerCase(Locale.ROOT);

            // Optional length input
            boolean nameLooksLikeLen = lname.equals("length") ||
                    lname.endsWith("_length") || lname.contains("length") ||
                    lname.contains("seq") || lname.contains("frames") || lname.contains("num");
            if (nameLooksLikeLen && (ti.type == OnnxJavaType.INT64 || ti.type == OnnxJavaType.INT32)) {
                if (sh.length == 0 || (sh.length == 1 && (sh[0] == 1 || sh[0] == -1))) {
                    foundLen = name;
                    lenIsI64 = (ti.type == OnnxJavaType.INT64);
                    continue;
                }
            }

            // Prefer 3D FLOAT features
            if (ti.type == OnnxJavaType.FLOAT && sh != null && sh.length == 3) {
                if ((sh[1] == DST_MELS) || (sh[2] == DST_MELS) ||
                        lname.contains("feature") || lname.contains("mel") || lname.contains("fbank")) {
                    foundData = name;
                    foundShape = sh.clone();
                    foundFeatures = true;
                    tmpMelAtDim1 = (sh[1] == DST_MELS) || (sh[1] == -1 && !lname.contains("time"));
                    continue;
                }
            }

            // Fallback: waveform (rank 1/2)
            if (foundData == null && ti.type == OnnxJavaType.FLOAT && (sh.length == 1 || sh.length == 2)) {
                foundData = name;
                foundShape = sh.clone();
                foundFeatures = false;
            }
        }

        if (foundData == null || foundShape == null) {
            throw new IllegalStateException("Unable to identify model data input (features or waveform).");
        }
        this.dataInputName = foundData;
        this.dataInputShape = foundShape;
        this.expectsFeatures = foundFeatures;
        this.lengthInputName = foundLen;
        this.lengthIsInt64 = lenIsI64;
        this.melAtDim1 = tmpMelAtDim1;

        if (expectsFeatures) {
            this.fbank = new Fbank(new Fbank.Config(this.sampleRate));

            Log.i(TAG, "Speakernet expects features. dataInput=" + dataInputName +
                    " shape=" + Arrays.toString(dataInputShape) +
                    (melAtDim1 ? " layout=[B,64,T]" : " layout=[B,T,64]") +
                    (lengthInputName != null ? (" lengthInput=" + lengthInputName +
                            " (" + (lengthIsInt64 ? "int64" : "int32") + ")") : ""));
        } else {
            this.fbank = null;
            Log.i(TAG, "Speakernet expects waveform. dataInput=" + dataInputName +
                    " shape=" + Arrays.toString(dataInputShape));
        }

        // Cache and log outputs once (constructor can throw OrtException)
        Map<String, NodeInfo> outInfo = session.getOutputInfo();
        this.outputNames = new ArrayList<>(outInfo.keySet());
        for (Map.Entry<String, NodeInfo> e : outInfo.entrySet()) {
            String name = e.getKey();
            NodeInfo ni = e.getValue();
            if (ni.getInfo() instanceof TensorInfo) {
                TensorInfo ti = (TensorInfo) ni.getInfo();
                Log.i(TAG, "Output " + name + " shape=" + Arrays.toString(ti.getShape()) + " type=" + ti.type);
            }
        }
    }

    public void resetStream() { pcm.clear(); finished = false; }

    /** Append PCM16 audio (short). */
    public void acceptWaveform(short[] i16) {
        if (i16 == null || i16.length == 0 || finished) return;
        for (short s : i16) pcm.add(s / 32768.0f);
    }

    /** Append PCM32 audio (float in [-1,1]). */
    public void acceptWaveform(float[] f32) {
        if (f32 == null || f32.length == 0 || finished) return;
        for (float v : f32) {
            float c = v;
            if (c > 1f) c = 1f;
            if (c < -1f) c = -1f;
            pcm.add(c);
        }
    }

    public void inputFinished() { finished = true; }

    /** Run one forward pass; returns L2-normalized embedding or empty array on failure. */
    public float[] computeEmbedding() throws OrtException {
        if (pcm.isEmpty()) return new float[0];

        OnnxTensor inData = null;
        OnnxTensor inLength = null;
        OrtSession.Result out = null;

        try {
            // ----- Build input -----
            if (expectsFeatures) {
                // PCM -> int16 -> FBANK [T,80]
                short[] i16 = new short[pcm.size()];
                for (int i = 0; i < i16.length; i++) {
                    float v = pcm.get(i);
                    if (v > 1f) v = 1f; if (v < -1f) v = -1f;
                    i16[i] = (short) Math.round(v * 32767f);
                }
                float[][] mels80 = fbank.compute(i16);
                if (mels80.length == 0) return new float[0];

                // If model wants 64 mels, take first 64 from 80
                float[][] mels = modelWants64Mels(dataInputShape) ? takeFirstMels(mels80, DST_MELS) : mels80;

                final int Torig = mels.length;
                final int Dmel  = (Torig > 0 ? mels[0].length : 0);

                // Fixed T?
                int requiredT = requiredFramesFromShape(dataInputShape, melAtDim1);
                int Tfeed = (requiredT > 0) ? requiredT : Torig;
                if (requiredT > 0 && Torig != requiredT) {
                    Log.w(TAG, String.format(Locale.US, "[EMB] adjusting frames %d→%d to match model", Torig, requiredT));
                }
                float[][] melsFeed = (Tfeed == Torig) ? mels : loopPadOrTrunc(mels, Tfeed);

                // Pack to expected layout
                float[] flat;
                long[] feedShape;
                if (melAtDim1) {
                    // [1, 64, T]
                    flat = new float[Dmel * Tfeed];
                    int pos = 0;
                    for (int mel = 0; mel < Dmel; mel++) {
                        for (int t = 0; t < Tfeed; t++) flat[pos++] = melsFeed[t][mel];
                    }
                    feedShape = new long[]{1, Dmel, Tfeed};
                } else {
                    // [1, T, 64]
                    flat = new float[Tfeed * Dmel];
                    int pos = 0;
                    for (int t = 0; t < Tfeed; t++) {
                        System.arraycopy(melsFeed[t], 0, flat, pos, Dmel);
                        pos += Dmel;
                    }
                    feedShape = new long[]{1, Tfeed, Dmel};
                }
                inData = OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), feedShape);

                // Optional length input (valid frames)
                if (lengthInputName != null) {
                    int valid = Math.min(Torig, Tfeed);
                    if (lengthIsInt64) {
                        inLength = OnnxTensor.createTensor(env, LongBuffer.wrap(new long[]{valid}), new long[]{1});
                    } else {
                        inLength = OnnxTensor.createTensor(env, IntBuffer.wrap(new int[]{valid}), new long[]{1});
                    }
                }
            } else {
                // Waveform: [1,T] or [T]
                float[] audio = new float[pcm.size()];
                for (int i = 0; i < pcm.size(); ++i) audio[i] = pcm.get(i);
                if (dataInputShape.length == 2) {
                    inData = OnnxTensor.createTensor(env, FloatBuffer.wrap(audio), new long[]{1, audio.length});
                } else {
                    inData = OnnxTensor.createTensor(env, FloatBuffer.wrap(audio), new long[]{audio.length});
                }
            }

            Map<String, OnnxTensor> feed = new HashMap<>();
            feed.put(dataInputName, inData);
            if (inLength != null) feed.put(lengthInputName, inLength);

            // ----- Run & pick the right output (embedding), not logits -----
            out = session.run(feed);
            float[] emb = pickEmbeddingOutput(out);
            if (emb.length == 0) return emb;

            l2normInPlace(emb);
            return emb;
        } catch (Throwable t) {
            Log.e(TAG, "computeEmbedding failed: " + t);
            return new float[0];
        } finally {
            if (out != null) try { out.close(); } catch (Exception ignore) {}
            if (inData != null) try { inData.close(); } catch (Exception ignore) {}
            if (inLength != null) try { inLength.close(); } catch (Exception ignore) {}
        }
    }

    /** Cosine similarity (L2-normalized inputs assumed). */
    public static float cosine(float[] a, float[] b) {
        int n = Math.min(a.length, b.length);
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += (double)a[i] * (double)b[i];
        return (float)s;
    }

    private static void l2normInPlace(float[] v) {
        double ss = 1e-10;
        for (float x : v) ss += (double)x * x;
        float inv = (float)(1.0 / Math.sqrt(ss));
        for (int i = 0; i < v.length; ++i) v[i] *= inv;
    }

    // ---------------------- Output selection (robust) ----------------------

    private float[] pickEmbeddingOutput(OrtSession.Result out) {
        // 1) Prefer names that look like embeddings
        String[] prefer = new String[]{"embs", "emb", "spk", "speaker"};
        for (int i = 0; i < outputNames.size(); i++) {
            String name = outputNames.get(i).toLowerCase(Locale.ROOT);
            boolean looksLikeEmb = false;
            for (String p : prefer) { if (name.contains(p)) { looksLikeEmb = true; break; } }
            if (!looksLikeEmb) continue;

            float[] cand = extractAsVector(out.get(i));
            if (cand.length >= 64 && cand.length <= 1024) {
                Log.d(TAG, "pickEmbeddingOutput: preferred by name '" + outputNames.get(i) + "', D=" + cand.length);
                return cand;
            }
        }

        // 2) Fallback: pick vector whose D is closest to 256 (as before)
        float[] best = new float[0];
        int bestScore = Integer.MAX_VALUE;
        for (int i = 0; i < outputNames.size(); i++) {
            float[] cand = extractAsVector(out.get(i));
            if (cand.length == 0) continue;
            int D = cand.length;
            if (D < 64 || D > 1024) continue;
            int score = Math.abs(D - 256);
            if (score < bestScore) { best = cand; bestScore = score; }
        }

        if (best.length == 0) {
            best = extractAsVector(out.get(0));
            if (best.length != 0) {
                Log.w(TAG, "pickEmbeddingOutput: fell back to output[0] length=" + best.length);
            } else {
                Log.e(TAG, "pickEmbeddingOutput: no usable float output found");
            }
        } else {
            Log.d(TAG, "pickEmbeddingOutput: chose by D~256, length=" + best.length);
        }
        return best;
    }

    /**
     * Convert an OnnxValue into a single [D] float vector:
     *   - float[]         -> [D]
     *   - float[][]       -> [B,D] -> row 0
     *   - float[][][]     -> [B,T,D] -> mean over T (row 0)
     * Returns empty array if unsupported.
     */
    private static float[] extractAsVector(OnnxValue val) {
        try {
            if (!(val instanceof OnnxTensor)) return new float[0];
            Object o = ((OnnxTensor) val).getValue();

            if (o instanceof float[]) {
                return (float[]) o;
            }

            if (o instanceof float[][]) {
                float[][] a = (float[][]) o;
                return (a.length > 0) ? a[0] : new float[0];
            }

            if (o instanceof float[][][]) {
                float[][][] a = (float[][][]) o;
                if (a.length == 0 || a[0].length == 0) return new float[0];
                float[][] bt = a[0]; // [T,D]
                int T = bt.length;
                int D = bt[0].length;
                float[] mean = new float[D];
                for (int t = 0; t < T; t++) {
                    float[] row = bt[t];
                    for (int d = 0; d < D; d++) mean[d] += row[d];
                }
                for (int d = 0; d < D; d++) mean[d] /= Math.max(1, T);
                return mean;
            }
        } catch (Throwable t) {
            Log.w(TAG, "extractAsVector: " + t);
        }
        return new float[0];
    }

    // ---------------------- Helpers ----------------------

    /** True if model declares 64 mel bins somewhere on the feature tensor. */
    private static boolean modelWants64Mels(long[] shape) {
        if (shape == null || shape.length != 3) return false;
        return (shape[1] == DST_MELS) || (shape[2] == DST_MELS);
    }

    /** If the model declares a fixed T, return it; otherwise -1. */
    private static int requiredFramesFromShape(long[] shape, boolean melAtDim1) {
        if (shape == null || shape.length != 3) return -1;
        long T = melAtDim1 ? shape[2] : shape[1];
        return (T > 0) ? (int) T : -1;
    }

    /** Take first K mel bins from [T, SRC_MELS] → [T, K]. */
    private static float[][] takeFirstMels(float[][] src, int k) {
        int T = src.length;
        if (T == 0) return src;
        int D = Math.min(k, src[0].length);
        float[][] out = new float[T][D];
        for (int t = 0; t < T; t++) System.arraycopy(src[t], 0, out[t], 0, D);
        return out;
    }

    /** Loop-pad or truncate a [T, D] mel matrix to exactly Tdst frames. */
    private static float[][] loopPadOrTrunc(float[][] src, int Tdst) {
        int Torig = src.length;
        int D = (Torig > 0) ? src[0].length : 0;
        float[][] out = new float[Tdst][D];
        if (Torig == 0 || D == 0) return out;

        if (Torig >= Tdst) {
            for (int t = 0; t < Tdst; t++) System.arraycopy(src[t], 0, out[t], 0, D);
        } else {
            int t = 0;
            while (t < Tdst) {
                int take = Math.min(Torig, Tdst - t);
                for (int i = 0; i < take; i++) System.arraycopy(src[i], 0, out[t + i], 0, D);
                t += take;
            }
        }
        return out;
    }

    @Override public void close() {
        try { session.close(); } catch (Exception ignore) {}
    }
}
