package ai.perplexity.hotword.speakerid;

import ai.onnxruntime.*;
import android.util.Log;

import java.io.*;
import java.util.*;

/**
 * Implements the Python logic: VAD-only segmentation, cluster enroll, running-mean target,
 * FLEX best-of scoring, and optional online adaptation.
 */
public final class SpeakerIdEngine implements AutoCloseable {
    private static final String TAG = "SpeakerIdEngine";

    private final SpeakerEmbedderOrt embedder;
    private final Vad vad;
    final SpeakerIdConfig cfg; // keep package-visible for Api to adjust K if needed

    // segmentation state
    private enum State { IDLE, ACTIVE }
    private State state = State.IDLE;
    private final int vadChunk; // samples
    private final int silenceAfterSamps;
    private final Deque<short[]> preroll = new ArrayDeque<>();
    private int accumulatedSilence = 0;
    private final ShortArray full = new ShortArray();
    private final ShortArray voiced = new ShortArray();

    // adaptation
    private float[] meanVec = null;
    private int meanCount = 0;
    private int addedThisRun = 0;

    // cluster
    private float[][] cluster = null; // K x D

    // Convenience passthrough for RN calls
    public float[] embedOnce(short[] seg) throws Exception {
        return embedFromI16(seg);
    }

    public SpeakerIdEngine(OrtEnvironment env,
                           OrtSession.SessionOptions options,
                           String speakernetOnnxPath,
                           Vad vad,
                           SpeakerIdConfig cfg) throws OrtException {
        this.embedder = new SpeakerEmbedderOrt(env, speakernetOnnxPath, options, cfg.rateHz);
        this.vad = vad;
        this.cfg = cfg.copy();
        this.vadChunk = cfg.vadChunk;
        this.silenceAfterSamps = (int) Math.round(cfg.silenceAfterSec * cfg.rateHz);

        // load persisted mean & count if present (with repair if needed)
        loadMeanIfExists();
        // load cluster if present
        loadClusterIfExists();
    }

    // ---------- Enrollment from a single utterance ----------
    public OnboardingResult enrollFromUtterance(short[] pcm) throws Exception {
        // 1) Segment and take the first voiced segment
        List<short[]> segs = segmentOffline(pcm);
        if (segs.isEmpty()) throw new IllegalStateException("No voiced segment found.");
        short[] seg = segs.get(0);

        // 2) Slice settings
        int win = secondsToSamps(cfg.sliceSec);
        int hop = secondsToSamps((cfg.sliceHopSec != null) ? cfg.sliceHopSec : cfg.sliceSec);

        // 3) Debug: show all slice windows (start/duration)
        List<short[]> slices = sliceI16(seg, win, hop);
        if (cfg.debugVadFrames) {
            int off = 0, idx = 0;
            for (short[] s : slices) {
                int startMs = (int) (1000L * off / cfg.rateHz);
                int durMs   = (int) (1000L * s.length / cfg.rateHz);
                Log.d(TAG, String.format(
                        java.util.Locale.US,
                        "[ONBOARD] slice#%d start_ms=%d dur_ms=%d samp=%d",
                        idx++, startMs, durMs, s.length));
                off += hop;
            }
            int voicedMs = (int)(1000L * seg.length / cfg.rateHz);
            Log.d(TAG, String.format(java.util.Locale.US,
                    "[ONBOARD] voiced_ms_total=%d slices_total=%d win_ms=%d hop_ms=%d",
                    voicedMs, slices.size(),
                    (int)(1000L * win / cfg.rateHz),
                    (int)(1000L * hop / cfg.rateHz)));
        }

        // 4) Embed each valid slice
        List<float[]> embs = sliceEmbeddings(seg, win, hop);
        if (embs.isEmpty()) {
            // Fallback: embed whole voiced segment (embedder will pad if needed)
            embs = java.util.Collections.singletonList(embedFromI16(seg));
        }

        // 5) Choose top-K (first K like python ref) → build cluster
        int totalSlices = embs.size();
        int K = Math.min(cfg.clusterSize, totalSlices);
        for (int i = 0; i < K; i++) {
            Log.d(TAG, String.format(java.util.Locale.US, "[ONBOARD] USED slice#%d", i));
        }
        float usedPct = (totalSlices > 0) ? (100f * K / totalSlices) : 0f;
        Log.i(TAG, String.format(java.util.Locale.US,
                "[ONBOARD] using %d/%d slices (%.1f%%) for cluster",
                K, totalSlices, usedPct));

        float[][] cl = new float[K][];
        for (int i = 0; i < K; ++i) {
            float[] e = embs.get(i);
            l2normInPlace(e);
            cl[i] = e;
        }
        final int D = cl[0].length;

        // 6) Save cluster + mean (+count) — mean saved as [1,D] row matrix
        try {
            NpyUtil.saveMatrixFloat32Atomic(cfg.clusterNpy, cl);
        } catch (NoSuchMethodError | UnsupportedOperationException ignore) {
            NpyUtil.saveMatrixFloat32(cfg.clusterNpy, cl);
        }

        float[] mean = meanOfRows(cl);
        l2normInPlace(mean);
        saveMeanAsRowMatrix(mean);   // robust write
        writeCountSidecar(cfg.meanEmbNpy, K);

        // 7) Verify end-to-end by reloading (robust)
        float[][] cl2 = tryLoadClusterMatrixWithRepair(cfg.clusterNpy, K, D);
        if (cl2.length != K || cl2[0].length != D) {
            throw new IOException("Cluster shape mismatch after write: got " +
                    cl2.length + "x" + cl2[0].length + " expected " + K + "x" + D);
        }
        for (float[] r : cl2) l2normInPlace(r); // ensure unit

        float[] mean2 = loadMeanFlexible(cfg.meanEmbNpy, D); // matrix-first, then vector, rewrite if needed
        if (mean2.length != D) {
            throw new IOException("enrollFromUtterance() Mean dim mismatch after write: " + mean2.length + " vs " + D);
        }
        for (float x : mean2) {
            if (!Float.isFinite(x)) throw new IOException("Mean contains non-finite");
        }

        // 8) Commit to memory
        this.cluster = cl2;
        this.meanVec = mean2;
        this.meanCount = K;

        Log.i(TAG, "[ONBOARD] success: K=" + K + " D=" + D +
                " files={" + cfg.clusterNpy + ", " + cfg.meanEmbNpy + "} verified OK");

        return new OnboardingResult(K, D);
    }

    // ---------- WWD: verify one voiced chunk (exact 1s) ----------
    /** Public helper for wake-word: verify exactly one voiced segment (e.g., ~1.0s). */
    public VerificationResult verifyFromVoicedSegment(short[] voicedSeg) throws Exception {
        if (voicedSeg == null || voicedSeg.length < secondsToSamps(cfg.minEmbedSec)) {
            return null;
        }
        // fullSeg==voicedSeg here
        return scoreAndMaybeAdapt(voicedSeg, voicedSeg);
    }

    // ---------- WWD: enroll directly from precomputed 1s embeddings ----------
    /** Public helper for wake-word: enroll from a list of embeddings (each already L2-normalized). */
    public OnboardingResult enrollFromEmbeddings(List<float[]> embs) throws Exception {
        if (embs == null || embs.isEmpty()) {
            throw new IllegalStateException("No embeddings provided.");
        }
        int K = Math.min(cfg.clusterSize, embs.size());
        float[][] cl = new float[K][];
        for (int i = 0; i < K; i++) {
            float[] e = Arrays.copyOf(embs.get(i), embs.get(i).length);
            l2normInPlace(e);
            cl[i] = e;
        }
        final int D = cl[0].length;

        try {
            NpyUtil.saveMatrixFloat32Atomic(cfg.clusterNpy, cl);
        } catch (NoSuchMethodError | UnsupportedOperationException ignore) {
            NpyUtil.saveMatrixFloat32(cfg.clusterNpy, cl);
        }

        float[] mean = meanOfRows(cl);
        l2normInPlace(mean);
        saveMeanAsRowMatrix(mean);
        writeCountSidecar(cfg.meanEmbNpy, K);

        // reload & verify
        float[][] cl2 = tryLoadClusterMatrixWithRepair(cfg.clusterNpy, K, D);
        if (cl2.length != K || cl2[0].length != D) {
            throw new IOException("Cluster shape mismatch after write: got " +
                    cl2.length + "x" + cl2[0].length + " expected " + K + "x" + D);
        }
        for (float[] r : cl2) l2normInPlace(r);

        float[] mean2 = loadMeanFlexible(cfg.meanEmbNpy, D);
        if (mean2.length != D) {
            throw new IOException("enrollFromEmbeddings() Mean dim mismatch after write: " + mean2.length + " vs " + D);
        }

        this.cluster = cl2;
        this.meanVec = mean2;
        this.meanCount = K;

        Log.i(TAG, "[ONBOARD/WWD] success: K=" + K + " D=" + D);
        return new OnboardingResult(K, D);
    }

    // ---------- Verification (streaming) ----------
    /** Push PCM16; returns a result when an utterance finalizes, else null. */
    public VerificationResult pushVerify(short[] pcm16) throws Exception {
        if (pcm16 == null || pcm16.length == 0) return null;

        int i = 0;
        while (i < pcm16.length) {
            int take = Math.min(vadChunk, pcm16.length - i);
            short[] block = Arrays.copyOfRange(pcm16, i, i + take);
            i += take;

            if (block.length < vadChunk) {
                block = padTo(block, vadChunk); // pad for VAD
            }
            float p = vad.feed(block);

            if (state == State.IDLE) {
                // preroll queue
                preroll.addLast(block);
                while (preroll.size() > cfg.prerollFrames) preroll.removeFirst();
                if (p >= cfg.onThr) {
                    state = State.ACTIVE;
                    // prepend preroll to both full & voiced
                    concatDequeInto(full, preroll);
                    concatDequeInto(voiced, preroll);
                    full.append(block);
                    voiced.append(block);
                    accumulatedSilence = 0;
                }
            } else {
                full.append(block);
                if (p >= cfg.offThr) {
                    voiced.append(block);
                    accumulatedSilence = 0;
                } else {
                    accumulatedSilence += vadChunk;
                    int minEmbed = secondsToSamps(cfg.minEmbedSec);
                    // grace window for short follow-up
                    int softLimit = silenceAfterSamps;
                    int hardLimit = silenceAfterSamps * 2;
                    int limit = (voiced.size() < minEmbed) ? hardLimit : softLimit;

                    if (accumulatedSilence >= limit) {
                        // finalize a segment
                        short[] fullSeg = full.toArray();
                        short[] voicedSeg = voiced.toArray();
                        resetSegState();
                        if (voicedSeg.length >= minEmbed) {
                            return scoreAndMaybeAdapt(fullSeg, voicedSeg);
                        }
                    }
                }
            }
        }
        return null;
    }

    /** Flush any active segment at end-of-stream. */
    public VerificationResult finishVerify() throws Exception {
        if (state == State.ACTIVE) {
            short[] fullSeg = full.toArray();
            short[] voicedSeg = voiced.toArray();
            resetSegState();
            if (voicedSeg.length >= secondsToSamps(cfg.minEmbedSec)) {
                return scoreAndMaybeAdapt(fullSeg, voicedSeg);
            }
        }
        return null;
    }

    // ---------- Core scoring / FLEX (multi-target, verbose) ----------
    private VerificationResult scoreAndMaybeAdapt(short[] fullSeg, short[] voicedSeg) throws Exception {
        float fullSec = fullSeg.length / (float) cfg.rateHz;
        float voicedSec = voicedSeg.length / (float) cfg.rateHz;

        // Targets: [mean] + cluster rows
        List<float[]> targets = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        ensureMeanLoaded(); // will also repair wrong-shaped mean on load
        targets.add(meanVec); labels.add("mean");
        if (cluster != null) {
            for (int i = 0; i < cluster.length; ++i) {
                targets.add(cluster[i]); labels.add("c#" + (i + 1));
            }
        }

        FlexEval out = flexEvalMultiVerbose(voicedSeg, targets, labels);
        // Online adaptation: add winning segment if above threshold
        if (cfg.addSampleThreshold >= 0 && out.bestScore >= cfg.addSampleThreshold && addedThisRun < cfg.addSampleMax) {
            float[] newEmb = embedFromI16(out.bestSegment);
            addToRunningMean(newEmb);
            addedThisRun += 1;
        }
        return new VerificationResult(fullSec, voicedSec, out.bestScore, out.bestStrategy, out.bestTargetLabel,
                out.perTargetStrategy);
    }

    // Holds verbose FLEX evaluation
    private static final class FlexEval {
        float bestScore = Float.NEGATIVE_INFINITY;
        String bestStrategy = "none";
        String bestTargetLabel = "none";
        short[] bestSegment = null;
        Map<String, Map<String, Float>> perTargetStrategy = new LinkedHashMap<>();
    }

    private FlexEval flexEvalMultiVerbose(short[] voicedSeg, List<float[]> targets, List<String> labels) throws Exception {
        FlexEval fe = new FlexEval();
        if (voicedSeg.length < secondsToSamps(cfg.minEmbedSec)) return fe;

        // normalized target matrix D x M
        float[][] T = new float[targets.size()][];
        for (int i = 0; i < targets.size(); ++i) T[i] = l2copy(targets.get(i));
        // helper: score a list of slices → best per target, + keep segment for _max strategies
        class Scored {
            final String tag; final List<short[]> slices; final float[][] scores; // S x M
            Scored(String tag, List<short[]> slices, float[][] scores) { this.tag = tag; this.slices = slices; this.scores = scores; }
        }

        List<Scored> items = new ArrayList<>();

        // base slices
        List<short[]> base = sliceI16(voicedSeg, secondsToSamps(cfg.sliceSec),
                secondsToSamps(cfg.sliceHopSec != null ? cfg.sliceHopSec : cfg.sliceSec));
        if (!base.isEmpty()) items.add(new Scored("base", base, scoreSlices(base, T)));

        // multi-res
        if (cfg.flexEnabled) {
            for (float s : cfg.flexSizesSec) {
                int win = secondsToSamps(s);
                int hop = Math.max(1, win / 2);
                List<short[]> sl = sliceI16(voicedSeg, win, hop);
                if (!sl.isEmpty()) items.add(new Scored(String.format(Locale.US, "mr%.2f", s), sl, scoreSlices(sl, T)));
            }
        }

        // whole
        int maxSamps = secondsToSamps(cfg.flexMaxSec);
        short[] whole = Arrays.copyOf(voicedSeg, Math.min(voicedSeg.length, maxSamps));
        if (whole.length < secondsToSamps(cfg.minEmbedSec)) {
            whole = loopOrPad(whole, secondsToSamps(Math.max(cfg.minEmbedSec, 0.5f)));
        }
        List<short[]> wlist = Collections.singletonList(whole);
        items.add(new Scored("whole", wlist, scoreSlices(wlist, T)));

        // Aggregate per-target metrics and pick the global best
        for (Scored sc : items) {
            float[][] mat = sc.scores; // S x M
            int S = mat.length, M = mat[0].length;
            // max per target
            Map<String, Float> perTmax = new LinkedHashMap<>();
            float globalMax = Float.NEGATIVE_INFINITY; int globalMaxTi = -1; int globalMaxSi = -1;
            for (int ti = 0; ti < M; ++ti) {
                float best = Float.NEGATIVE_INFINITY; int bestSi = -1;
                for (int si = 0; si < S; ++si) {
                    if (mat[si][ti] > best) { best = mat[si][ti]; bestSi = si; }
                }
                perTmax.put(labels.get(ti), best);
                if (best > globalMax) { globalMax = best; globalMaxTi = ti; globalMaxSi = bestSi; }
            }
            fe.perTargetStrategy.put(sc.tag + "_max", perTmax);
            if (globalMax > fe.bestScore) {
                fe.bestScore = globalMax;
                fe.bestStrategy = sc.tag + "_max";
                fe.bestTargetLabel = labels.get(globalMaxTi);
                fe.bestSegment = sc.slices.get(globalMaxSi);
            }
            // top-k mean per target
            int k = Math.min(cfg.flexTopK, S);
            if (k >= 2) {
                Map<String, Float> perTtopk = new LinkedHashMap<>();
                for (int ti = 0; ti < M; ++ti) {
                    float[] col = new float[S]; for (int si = 0; si < S; ++si) col[si] = mat[si][ti];
                    Arrays.sort(col);
                    float sum = 0f; int kk = 0;
                    for (int idx = S - 1; idx >= Math.max(0, S - k); --idx) { sum += col[idx]; kk++; }
                    perTtopk.put(labels.get(ti), kk > 0 ? sum / kk : Float.NEGATIVE_INFINITY);
                }
                fe.perTargetStrategy.put(sc.tag + "_top" + k + "_mean", perTtopk);
            }
        }
        return fe;
    }

    private float[][] scoreSlices(List<short[]> slices, float[][] targetsColMajor) throws Exception {
        int S = slices.size(), M = targetsColMajor.length;
        float[][] out = new float[S][M];
        for (int i = 0; i < S; ++i) {
            float[] emb = embedFromI16(slices.get(i));
            for (int ti = 0; ti < M; ++ti) {
                out[i][ti] = SpeakerEmbedderOrt.cosine(emb, targetsColMajor[ti]);
            }
        }
        return out;
    }

    private java.util.List<float[]> sliceEmbeddings(short[] voicedSeg, int win, int hop) throws Exception {
        java.util.ArrayList<float[]> out = new java.util.ArrayList<>();
        if (voicedSeg == null || voicedSeg.length == 0 || win <= 0) return out;

        int i = 0;
        while (i + win <= voicedSeg.length) {
            short[] sl = java.util.Arrays.copyOfRange(voicedSeg, i, i + win);
            out.add(embedFromI16(sl));
            i += hop;
        }
        int rem = voicedSeg.length - i;
        if (rem > 0) {
            short[] tail = Arrays.copyOfRange(voicedSeg, i, voicedSeg.length);
            out.add(embedFromI16(tail));
        }
        return out;
    }

    private List<short[]> sliceI16(short[] x, int win, int hop) {
        ArrayList<short[]> out = new ArrayList<>();
        if (x.length == 0 || win <= 0) return out;
        int i = 0;
        while (i + win <= x.length) {
            out.add(Arrays.copyOfRange(x, i, i + win));
            i += hop;
        }
        int rem = x.length - i;
        if (rem > 0) {
            out.add(Arrays.copyOfRange(x, i, x.length));
        }
        return out;
    }

    // ---------- Segmentation helpers ----------
    private List<short[]> segmentOffline(short[] pcm) {
        List<short[]> segs = new ArrayList<>();
        ShortArray fullB = new ShortArray();
        ShortArray voicedB = new ShortArray();
        Deque<short[]> pr = new ArrayDeque<>();
        int silence = 0;
        State st = State.IDLE;
        int i = 0;
        while (i < pcm.length) {
            int take = Math.min(vadChunk, pcm.length - i);
            short[] block = Arrays.copyOfRange(pcm, i, i + take); i += take;
            if (block.length < vadChunk) block = padTo(block, vadChunk);
            float p = vad.feed(block);

            if (st == State.IDLE) {
                pr.addLast(block);
                while (pr.size() > cfg.prerollFrames) pr.removeFirst();
                if (p >= cfg.onThr) {
                    st = State.ACTIVE;
                    concatDequeInto(fullB, pr);
                    concatDequeInto(voicedB, pr);
                    fullB.append(block);
                    voicedB.append(block);
                    silence = 0;
                }
            } else {
                fullB.append(block);
                if (p >= cfg.offThr) {
                    voicedB.append(block);
                    silence = 0;
                } else {
                    silence += vadChunk;
                    if (silence >= silenceAfterSamps) {
                        segs.add(voicedB.toArray());
                        st = State.IDLE; pr.clear(); fullB.clear(); voicedB.clear(); silence = 0;
                    }
                }
            }
        }
        if (st == State.ACTIVE && voicedB.size() >= secondsToSamps(cfg.minEmbedSec))
            segs.add(voicedB.toArray());
        return segs;
    }

    private void resetSegState() {
        state = State.IDLE;
        preroll.clear();
        full.clear();
        voiced.clear();
        accumulatedSilence = 0;
    }

    private static short[] padTo(short[] x, int n) {
        short[] y = new short[n];
        System.arraycopy(x, 0, y, 0, x.length);
        return y;
    }
    private static void concatDequeInto(ShortArray dst, Deque<short[]> q) {
        for (short[] b : q) dst.append(b);
    }

    private static void ensureFinite(float[] v, String tag) {
        for (float x : v) if (!Float.isFinite(x))
            throw new IllegalStateException(tag + " has non-finite values");
    }

    private float[] embedFromI16(short[] seg) throws Exception {
        seg = ensureMinSamplesForModel(seg); // ensures >= ~655 ms @16k

        embedder.resetStream();
        embedder.acceptWaveform(seg);
        embedder.inputFinished();
        float[] e = embedder.computeEmbedding();
        if (e == null || e.length == 0) throw new IllegalStateException("Empty embedding");
        ensureFinite(e, "embedding");
        l2normInPlace(e);
        return e;
    }

    private static float[] meanOfRows(float[][] m) {
        int rows = m.length, cols = m[0].length;
        float[] out = new float[cols];
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) out[c] += m[r][c];
        for (int c = 0; c < cols; ++c) out[c] /= Math.max(1, rows);
        return out;
    }
    private static float[] l2copy(float[] v) {
        float[] o = Arrays.copyOf(v, v.length);
        l2normInPlace(o); return o;
    }
    private static void l2normInPlace(float[] v) {
        double ss = 1e-10;
        for (float x : v) ss += (double)x * x;
        float inv = (float)(1.0 / Math.sqrt(ss));
        for (int i = 0; i < v.length; ++i) v[i] *= inv;
    }

    private static short[] loopOrPad(short[] x, int want) {
        if (x.length >= want) return Arrays.copyOf(x, want);
        if (x.length == 0) return new short[want];
        short[] y = new short[want]; int i = 0;
        while (i < want) {
            int take = Math.min(x.length, want - i);
            System.arraycopy(x, 0, y, i, take); i += take;
        }
        return y;
    }

    private int secondsToSamps(float s) { return (int)Math.round(s * cfg.rateHz); }

    private short[] ensureMinSamplesForModel(short[] x) {
        // For 16kHz, 25ms window, 10ms hop, 64 frames ≈ 25ms + 63*10ms ≈ 655ms
        int want = secondsToSamps(0.655f);
        if (x.length >= want) return x;
        short[] y = loopOrPad(x, want); // repeats to fill (or pads zeros if empty)
        if (cfg.debugVadFrames) {
            Log.w(TAG, String.format(java.util.Locale.US,
                    "[EMB] padded slice from %d→%d samp (~%d ms) to satisfy 64-frame min",
                    x.length, y.length, (int)(1000L * y.length / cfg.rateHz)));
        }
        return y;
    }

    // ---------- Persistence (mean & cluster) ----------
    private void ensureMeanLoaded() throws IOException {
        if (meanVec != null) return;
        if (cfg.meanEmbNpy != null && cfg.meanEmbNpy.exists()) {
            float[] v = loadMeanFlexible(cfg.meanEmbNpy, /*expectedD*/ -1);
            meanVec = v;
            meanCount = readCountSidecar(cfg.meanEmbNpy, 1);
        } else {
            throw new IllegalStateException("Mean embedding not found. Run enrollment first.");
        }
    }

    private void loadMeanIfExists() {
        try {
            if (cfg.meanEmbNpy != null && cfg.meanEmbNpy.exists()) {
                if (cfg.meanEmbNpy.length() < 64) {
                    // clearly corrupt/tiny; wipe
                    //noinspection ResultOfMethodCallIgnored
                    cfg.meanEmbNpy.delete();
                    File cnt = new File(cfg.meanEmbNpy.getAbsolutePath() + ".count");
                    if (cnt.exists()) //noinspection ResultOfMethodCallIgnored
                        cnt.delete();
                    Log.w(TAG, "Mean emb file too small; deleted.");
                    return;
                }
                meanVec = loadMeanFlexible(cfg.meanEmbNpy, /*expectedD*/ -1);
                meanCount = readCountSidecar(cfg.meanEmbNpy, 1);
            }
        } catch (Exception e) {
            Log.w(TAG, "Failed to load mean emb; wiping. " + e);
            try {
                //noinspection ResultOfMethodCallIgnored
                cfg.meanEmbNpy.delete();
                File cnt = new File(cfg.meanEmbNpy.getAbsolutePath() + ".count");
                if (cnt.exists()) //noinspection ResultOfMethodCallIgnored
                    cnt.delete();
            } catch (Exception ignore) {}
            meanVec = null;
            meanCount = 0;
        }
    }

    private void loadClusterIfExists() {
        try {
            if (cfg.clusterNpy != null && cfg.clusterNpy.exists()) {
                if (cfg.clusterNpy.length() < 64) {
                    //noinspection ResultOfMethodCallIgnored
                    cfg.clusterNpy.delete();
                    Log.w(TAG, "Cluster file too small; deleted.");
                    cluster = null;
                    return;
                }
                cluster = NpyUtil.loadMatrixFloat32(cfg.clusterNpy);
                for (float[] r : cluster) l2normInPlace(r);
            }
        } catch (Exception e) {
            Log.w(TAG, "Failed to load cluster; wiping. " + e);
            try { //noinspection ResultOfMethodCallIgnored
                cfg.clusterNpy.delete();
            } catch (Exception ignore) {}
            cluster = null;
        }
    }

    /** Write mean as [1,D] (row matrix) for robust reload across loaders. */
    private void saveMeanAsRowMatrix(float[] mean) throws IOException {
        float[][] row = new float[1][mean.length];
        System.arraycopy(mean, 0, row[0], 0, mean.length);
        try {
            NpyUtil.saveMatrixFloat32Atomic(cfg.meanEmbNpy, row);
        } catch (NoSuchMethodError | UnsupportedOperationException ignore) {
            NpyUtil.saveMatrixFloat32(cfg.meanEmbNpy, row);
        }
    }

    /** Robust mean reload: matrix-first (flatten), then vector; rewrite to vector if needed. */
    private float[] loadMeanFlexible(File meanFile, int expectedD) throws IOException {
        // 1) Matrix first
        try {
            float[][] m = NpyUtil.loadMatrixFloat32(meanFile);
            if (m != null && m.length == 1 && m[0] != null) {
                float[] flat = Arrays.copyOf(m[0], m[0].length);
                if (expectedD > 0 && flat.length != expectedD) {
                    throw new IOException("loadMeanFlexible() First mean dim mismatch after write (matrix): " + flat.length + " vs " + expectedD);
                }
                // Rewrite to 1-D vector for future fast loads
                try {
                    NpyUtil.saveVectorFloat32Atomic(meanFile, flat);
                } catch (NoSuchMethodError | UnsupportedOperationException ignore) {
                    NpyUtil.saveVectorFloat32(meanFile, flat);
                }
                Log.w(TAG, "Rewrote mean from [1,D] to 1-D vector: " + meanFile);
                return flat;
            }
        } catch (Throwable ignore) {
            // fall through
        }

        // 2) Vector next
        float[] vec = null;
        try {
            vec = NpyUtil.loadVectorFloat32(meanFile);
        } catch (Throwable ignore) {}
        if (vec != null && (expectedD <= 0 || vec.length == expectedD)) return vec;

        // If buggy vector loader produced length==1, attempt matrix again before failing
        if (vec != null && vec.length == 1 && expectedD > 1) {
            try {
                float[][] m = NpyUtil.loadMatrixFloat32(meanFile);
                if (m != null && m.length == 1 && m[0] != null && m[0].length == expectedD) {
                    float[] flat = Arrays.copyOf(m[0], m[0].length);
                    try {
                        NpyUtil.saveVectorFloat32Atomic(meanFile, flat);
                    } catch (NoSuchMethodError | UnsupportedOperationException ignore) {
                        NpyUtil.saveVectorFloat32(meanFile, flat);
                    }
                    Log.w(TAG, "Recovered mean by reading matrix and rewriting vector: " + meanFile);
                    return flat;
                }
            } catch (Throwable ignore) {}
        }

        // FINAL RECOVERY: rebuild mean from cluster and rewrite as vector
        try {
            if (expectedD > 1 && cfg.clusterNpy != null && cfg.clusterNpy.exists()) {
                float[][] cl = NpyUtil.loadMatrixFloat32(cfg.clusterNpy);
                if (cl != null && cl.length > 0) {
                    int D = cl[0].length;
                    if (expectedD <= 0 || D == expectedD) {
                        float[] rebuilt = meanOfRows(cl);
                        l2normInPlace(rebuilt);
                        try { NpyUtil.saveVectorFloat32Atomic(meanFile, rebuilt); }
                        catch (NoSuchMethodError | UnsupportedOperationException ignore) { NpyUtil.saveVectorFloat32(meanFile, rebuilt); }
                        Log.w(TAG, "Rebuilt mean from cluster and rewrote as 1-D vector: " + meanFile);
                        return rebuilt;
                    }
                }
            }
        } catch (Throwable ignore) { /* fall through */ }

        // Final error
        int got = (vec == null ? -1 : vec.length);
        throw new IOException("loadMeanFlexible() Last Line Mean dim mismatch after write: " + got + " vs " + expectedD);
    }

    private static File countFile(File meanNpy) { return new File(meanNpy.getAbsolutePath() + ".count"); }
    private static void writeCountSidecar(File meanNpy, int count) throws IOException {
        try (Writer w = new OutputStreamWriter(new FileOutputStream(countFile(meanNpy)), "UTF-8")) {
            w.write(Integer.toString(count));
        }
    }
    private static int readCountSidecar(File meanNpy, int def) {
        File f = countFile(meanNpy);
        if (!f.exists()) return def;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f), "UTF-8"))) {
            return Integer.parseInt(br.readLine().trim());
        } catch (Exception e) {
            return def;
        }
    }

    /** Load a KxD matrix; if a legacy 1-D vector is found and K==1, reshape and rewrite. */
    private float[][] tryLoadClusterMatrixWithRepair(File clusterFile, int expectedK, int expectedD) throws IOException {
        float[][] m = NpyUtil.loadMatrixFloat32(clusterFile);
        if (m != null && m.length == expectedK && m[0] != null && m[0].length == expectedD) {
            return m; // all good
        }
        // If it came back suspicious (e.g., 1x1) try reading as a vector then reshape
        try {
            float[] v = NpyUtil.loadVectorFloat32(clusterFile);
            if (v != null && expectedK == 1 && v.length == expectedD) {
                float[][] fixed = new float[1][expectedD];
                System.arraycopy(v, 0, fixed[0], 0, expectedD);
                try {
                    NpyUtil.saveMatrixFloat32Atomic(clusterFile, fixed);
                } catch (NoSuchMethodError | UnsupportedOperationException ignore) {
                    NpyUtil.saveMatrixFloat32(clusterFile, fixed);
                }
                Log.w(TAG, "Rewrote legacy cluster (vector) to 2-D [1," + expectedD + "]: " + clusterFile);
                return fixed;
            }
        } catch (Throwable ignore) { /* fall through */ }
        return m; // caller will validate and throw if still wrong
    }

    private void addToRunningMean(float[] newEmbUnit) throws IOException {
        if (meanVec == null) {
            meanVec = Arrays.copyOf(newEmbUnit, newEmbUnit.length);
            meanCount = 1;
        } else {
            int n = meanCount;
            float[] out = new float[meanVec.length];
            for (int i = 0; i < out.length; ++i)
                out[i] = (meanVec[i] * n + newEmbUnit[i]) / (n + 1);
            l2normInPlace(out);
            meanVec = out; meanCount = n + 1;
        }

        // Persist as 1-D vector (fast path; readers handle both)
        try {
            NpyUtil.saveVectorFloat32Atomic(cfg.meanEmbNpy, meanVec);
        } catch (NoSuchMethodError | UnsupportedOperationException ignore) {
            NpyUtil.saveVectorFloat32(cfg.meanEmbNpy, meanVec);
        }

        writeCountSidecar(cfg.meanEmbNpy, meanCount);
        Log.i(TAG, "[ADAPT] Added sample → new_count=" + meanCount + " saved: " + cfg.meanEmbNpy);
    }

    /** <— THIS IS THE API YOUR SpeakerIdApi CALLS */
    public void resetTargetsInMemory() {
        this.meanVec = null;
        this.meanCount = 0;
        this.cluster = null;
        this.addedThisRun = 0;
        Log.i(TAG, "[RESET] Cleared in-memory mean/cluster.");
    }

    @Override public void close() {
        try { embedder.close(); } catch (Exception ignore) {}
    }

    // ---------- Tiny ShortArray helper ----------
    private static final class ShortArray {
        short[] a = new short[0]; int n = 0;
        void clear(){ n=0; }
        void append(short[] b){ ensure(n+b.length); System.arraycopy(b,0,a,n,b.length); n+=b.length; }
        short[] toArray(){ return java.util.Arrays.copyOf(a,n); }
        int size(){ return n; }
        private void ensure(int m){ if(a.length>=m)return; a=java.util.Arrays.copyOf(a, Math.max(m,a.length*2+1024)); }
    }
}
