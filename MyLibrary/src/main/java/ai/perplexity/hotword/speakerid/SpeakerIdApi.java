package ai.perplexity.hotword.speakerid;

import android.content.Context;
import android.net.Uri;
import android.util.Log;
import ai.onnxruntime.*;
import androidx.annotation.RequiresPermission;
import androidx.core.content.ContextCompat;
import android.Manifest;
import android.content.pm.PackageManager;

import java.io.*;
import java.nio.FloatBuffer;
import java.util.*;

/**
 * High-level API: onboarding (mic/stream/wav) + verification (mic/stream/wav),
 * with default storage/export utilities. Adds Wake-Word specific APIs (WWD) too.
 */
public final class SpeakerIdApi implements AutoCloseable {
    private static final String TAG = "SpeakerIdApi";

    private final SpeakerIdConfig cfg;
    private final Vad vad;
    private final SpeakerIdEngine engine;
    private final OrtEnvironment env;
    private final OrtSession.SessionOptions opts;
    private final Context appContext;

    // keep constructor as-is
    private SpeakerIdApi(Context ctx,
                        SpeakerIdConfig cfg,
                        Vad vad,
                        SpeakerIdEngine engine,
                        OrtEnvironment env,
                        OrtSession.SessionOptions opts) {
        this.appContext = ctx.getApplicationContext();
        this.cfg = cfg;
        this.vad = vad;
        this.engine = engine;
        this.env = env;
        this.opts = opts;
    }

    // Standard create() with normal config
    public static SpeakerIdApi create(Context ctx) throws OrtException {
        Log.i(TAG, "create()");
        SpeakerIdAssets.Paths p = SpeakerIdAssets.resolveModels(ctx);
        if (p.speakerOnnx == null) {
            String direct = SpeakerIdAssets.copyAssetIfExists(ctx,
                    "nemo_en_speakerverification_speakernet.onnx", "speakerid");
            if (direct != null) p = new SpeakerIdAssets.Paths(direct, p.vadOnnx);
        }
        if (p.speakerOnnx == null) {
            throw new IllegalStateException("Speakernet ONNX not found (speaker_id.dm or direct asset).");
        }

        // ORT env + options
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        opts.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors()/2));
        opts.setInterOpNumThreads(1);

        // Config (defaults + storage paths)
        SpeakerIdConfig cfg = new SpeakerIdConfig();
        cfg.meanEmbNpy = SpeakerIdStorage.defaultMeanEmbFile(ctx);
        cfg.clusterNpy = SpeakerIdStorage.defaultClusterFile(ctx);

        // VAD (if available)
        Vad vad;
        try {
            if (p.vadOnnx != null) {
                vad = new VadAdapter(ctx, p.vadOnnx);
            } else {
                Log.w(TAG, "No VAD model found; attempting silero_vad.onnx fallback.");
                vad = new VadAdapter(ctx, SpeakerIdAssets.copyAssetIfExists(ctx, "silero_vad.onnx", "speakerid"));
            }
        } catch (Throwable t) {
            Log.w(TAG, "VAD init failed, proceeding without VAD: " + t);
            vad = new Vad() { @Override public float feed(short[] b){ return 1.0f; } };
        }

        SpeakerIdEngine engine = new SpeakerIdEngine(env, opts, p.speakerOnnx, vad, cfg);
        return new SpeakerIdApi(ctx, cfg, vad, engine, env, opts);
    }

    public float[] embedOnce(short[] oneSecondPcm) throws Exception {
        return engine.embedOnce(oneSecondPcm); // L2-normalized
    }

    // --------- NEW: create a WWD-tuned instance (separate storage) ----------
    public static SpeakerIdApi createWWD(Context ctx) throws OrtException {
        Log.i(TAG, "createWWD()");
        SpeakerIdAssets.Paths p = SpeakerIdAssets.resolveModels(ctx);
        if (p.speakerOnnx == null) {
            String direct = SpeakerIdAssets.copyAssetIfExists(ctx,
                    "nemo_en_speakerverification_speakernet.onnx", "speakerid");
            if (direct != null) p = new SpeakerIdAssets.Paths(direct, p.vadOnnx);
        }
        if (p.speakerOnnx == null) {
            throw new IllegalStateException("Speakernet ONNX not found.");
        }

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        opts.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors()/2));
        opts.setInterOpNumThreads(1);

        // Use your WWD config and separate files
        SpeakerIdConfigWWD wwd = new SpeakerIdConfigWWD();
        SpeakerIdConfig cfg = wwd.copy();
        // Use WWD-specific filenames so they don't clash with the regular profile
        cfg.meanEmbNpy = new File(ctx.getFilesDir(), "speaker_emb_wwd.npy");
        cfg.clusterNpy = new File(ctx.getFilesDir(), "speaker_emb_cluster_wwd.npy");

        Vad vad;
        try {
            if (p.vadOnnx != null) {
                vad = new VadAdapter(ctx, p.vadOnnx);
            } else {
                Log.w(TAG, "No VAD model found; attempting silero_vad.onnx fallback.");
                vad = new VadAdapter(ctx, SpeakerIdAssets.copyAssetIfExists(ctx, "silero_vad.onnx", "speakerid"));
            }
        } catch (Throwable t) {
            Log.w(TAG, "VAD init failed, proceeding without VAD: " + t);
            vad = new Vad() { @Override public float feed(short[] b){ return 1.0f; } };
        }

        SpeakerIdEngine engine = new SpeakerIdEngine(env, opts, p.speakerOnnx, vad, cfg);
        return new SpeakerIdApi(ctx, cfg, vad, engine, env, opts);
    }

    // Update close():
    @Override public void close() {
        try { if (engine != null) engine.close(); } catch (Throwable ignore) {}
        try { if (vad instanceof java.io.Closeable) ((java.io.Closeable) vad).close(); } catch (Throwable ignore) {}
    }

    // SpeakerIdApi.java (add public method)
    public void wipeAllTargetsAndReset() {
        SpeakerIdStorage.wipeDefaults(appContext);
        engine.resetTargetsInMemory();
        Log.i(TAG, "[RESET] Disk files deleted and engine state cleared. Ready to re-onboard.");
    }

    /** Return true if default mean/cluster exist (good for verification init flow). */
    public boolean initVerificationUsingDefaults(Context ctx) {
        return SpeakerIdStorage.hasDefaultTargets(ctx);
    }

    /** Use explicit mean/cluster files for verification; returns success flag. */
    public boolean initVerificationWithFiles(File meanNpy, File clusterNpy) {
        try {
            if (!meanNpy.exists() || !clusterNpy.exists()) return false;
            cfg.meanEmbNpy = meanNpy;
            cfg.clusterNpy = clusterNpy;
            return true;
        } catch (Exception e) {
            Log.w(TAG, "initVerificationWithFiles failed: " + e.getMessage());
            return false;
        }
    }

    // ---------- Onboarding (MIC) ----------
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    public OnboardingResult onboardFromMicrophone(long maxMillis) throws Exception {
        return onboardFromMicrophoneUntil(cfg.onboardVoicedTargetSec, /*hardTimeoutMs*/ 0L);
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    public OnboardingResult onboardFromMicrophoneUntil(float targetVoicedSec, long hardTimeoutMs) throws Exception {
        if (ContextCompat.checkSelfPermission(appContext, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            throw new SecurityException("RECORD_AUDIO permission not granted. Request it before starting MicRecorder.");
        }
        long deadline = (hardTimeoutMs > 0) ? System.currentTimeMillis() + hardTimeoutMs : Long.MAX_VALUE;

        try (MicRecorder mic = new MicRecorder(appContext, cfg.rateHz, cfg.vadChunk)) {
            mic.start();
            short[] seg = collectVoicedSegmentFromMic(mic, targetVoicedSec, deadline);
            if (seg == null || seg.length < secondsToSamps(cfg.minEmbedSec)) {
                throw new IllegalStateException("Onboarding failed: collected voiced=" +
                        (seg == null ? 0 : seg.length) + " samp (" +
                        (seg == null ? 0 : (1000L * seg.length / cfg.rateHz)) + " ms).");
            }
            OnboardingResult r = engine.enrollFromUtterance(seg);
            Log.i(TAG, String.format(java.util.Locale.US,
                    "[ONBOARD/MIC] ok: voiced_ms=%d target_sec=%.2f seg_len=%d samp",
                    (int)(1000L * seg.length / cfg.rateHz), targetVoicedSec, seg.length));
            return r;
        }
    }

    private short[] collectVoicedSegmentFromMic(MicRecorder mic, float targetVoicedSec, long deadline) {
        final int vadChunk = cfg.vadChunk;
        final int silenceAfterSamps = (int)Math.round(cfg.silenceAfterSec * cfg.rateHz);
        final int targetVoicedSamps = (int)Math.round(targetVoicedSec * cfg.rateHz);

        java.util.ArrayDeque<short[]> preroll = new java.util.ArrayDeque<>();
        ShortArray voiced = new ShortArray();
        ShortArray full   = new ShortArray();

        boolean active = false;
        int silence = 0;
        long totalSamps = 0;

        while (System.currentTimeMillis() < deadline) {
            short[] block = mic.readBlock();               // returns <= vadChunk
            if (block == null) continue;
            if (block.length < vadChunk) {
                short[] pad = new short[vadChunk];
                System.arraycopy(block, 0, pad, 0, block.length);
                block = pad;
            }
            totalSamps += block.length;

            float p = vad.feed(block);
            String state = active ? "ACTIVE" : "IDLE";

            if (!active) {
                preroll.addLast(block);
                while (preroll.size() > cfg.prerollFrames) preroll.removeFirst();

                if (cfg.debugVadFrames) {
                    Log.d(TAG, String.format(java.util.Locale.US,
                        "[VAD] t=%.3fs p=%.3f st=%s action=%s voiced_ms=%d total_ms=%d",
                        totalSamps / (float)cfg.rateHz, p, state, "preroll",
                        (int)(1000L * voiced.n / cfg.rateHz), (int)(1000L * totalSamps / cfg.rateHz)));
                }

                if (p >= cfg.onThr) {
                    active = true; 
                    for (short[] pr : preroll) full.append(pr);
                    full.append(block);

                    // Start voiced strictly from the first ACTIVE block
                    voiced.append(block);
                    silence = 0;

                    if (cfg.debugVadFrames) {
                        Log.d(TAG, String.format(java.util.Locale.US,
                            "[VAD] ENTER ACTIVE p=%.3f preroll_frames=%d voiced_ms=%d",
                            p, preroll.size(), (int)(1000L * voiced.n / cfg.rateHz)));
                    }
                }
            } else {
                full.append(block);
                if (p >= cfg.offThr) {
                    voiced.append(block);
                    silence = 0;
                    if (cfg.debugVadFrames) {
                        Log.d(TAG, String.format(java.util.Locale.US,
                            "[VAD] KEEP ACTIVE p=%.3f add=Y voiced_ms=%d", p, (int)(1000L * voiced.n / cfg.rateHz)));
                    }
                    if (voiced.n >= targetVoicedSamps) {
                        Log.i(TAG, String.format(java.util.Locale.US,
                            "[VAD] TARGET REACHED voiced_ms=%d target_ms=%d total_ms=%d",
                            (int)(1000L * voiced.n / cfg.rateHz),
                            (int)(1000L * targetVoicedSamps / cfg.rateHz),
                            (int)(1000L * totalSamps / cfg.rateHz)));
                        return voiced.toArray();
                    }
                } else {
                    silence += vadChunk;
                    if (cfg.debugVadFrames) {
                        Log.d(TAG, String.format(java.util.Locale.US,
                            "[VAD] SOFT OFF p=%.3f add=N silence_ms=%d voiced_ms=%d",
                            p, (int)(1000L * silence / cfg.rateHz), (int)(1000L * voiced.n / cfg.rateHz)));
                    }
                    if (silence >= silenceAfterSamps) {
                        Log.i(TAG, String.format(java.util.Locale.US,
                            "[VAD] SEGMENT END voiced_ms=%d full_ms=%d (silence_ms=%d)",
                            (int)(1000L * voiced.n / cfg.rateHz),
                            (int)(1000L * full.n   / cfg.rateHz),
                            (int)(1000L * silence  / cfg.rateHz)));
                        return voiced.toArray();
                    }
                }
            }
        }

        Log.w(TAG, String.format(java.util.Locale.US,
            "[VAD] DEADLINE HIT voiced_ms=%d total_ms=%d target_ms=%d",
            (int)(1000L * voiced.n / cfg.rateHz),
            (int)(1000L * totalSamps / cfg.rateHz),
            (int)(1000L * targetVoicedSamps / cfg.rateHz)));
        return null;
    }

    // ---------- Onboarding (STREAM) ----------
    public static final class OnboardingStream {
        private final SpeakerIdEngine engine;
        private final SpeakerIdConfig cfg;
        private final Vad vad;

        // segmentation state (same as engine)
        private enum State { IDLE, ACTIVE }
        private State st = State.IDLE;
        private final int vadChunk;
        private final int silenceAfterSamps;
        private final java.util.Deque<short[]> preroll = new java.util.ArrayDeque<>();
        private final ShortArray voiced = new ShortArray();
        private int silence = 0;

        private OnboardingResult done = null;

        private OnboardingStream(SpeakerIdEngine engine, SpeakerIdConfig cfg, Vad vad) {
            this.engine = engine; this.cfg = cfg; this.vad = vad;
            this.vadChunk = cfg.vadChunk;
            this.silenceAfterSamps = (int)Math.round(cfg.silenceAfterSec * cfg.rateHz);
        }

        /** Feed a block (any length). Returns onboarding result when finished, else null. */
        public OnboardingResult feed(short[] block) throws Exception {
            if (done != null) return done;
            int i = 0;
            while (i < block.length) {
                int take = Math.min(vadChunk, block.length - i);
                short[] b = java.util.Arrays.copyOfRange(block, i, i + take); i += take;
                if (b.length < vadChunk) b = padTo(b, vadChunk);
                float p = vad.feed(b);

                if (st == State.IDLE) {
                    preroll.addLast(b);
                    while (preroll.size() > cfg.prerollFrames) preroll.removeFirst();
                    if (p >= cfg.onThr) {
                        st = State.ACTIVE;
                        for (short[] pr : preroll) voiced.append(pr);
                        voiced.append(b);
                        silence = 0;
                    }
                } else {
                    if (p >= cfg.offThr) {
                        voiced.append(b);
                        silence = 0;
                    } else {
                        silence += vadChunk;
                        if (silence >= silenceAfterSamps) {
                            short[] seg = voiced.toArray();
                            reset();
                            if (seg.length >= secondsToSamps(cfg.minEmbedSec)) {
                                done = engine.enrollFromUtterance(seg);
                                return done;
                            }
                        }
                    }
                }
            }
            return null;
        }

        /** Call at end of stream to flush any active segment. */
        public OnboardingResult finish() throws Exception {
            if (done != null) return done;
            if (st == State.ACTIVE) {
                short[] seg = voiced.toArray();
                reset();
                if (seg.length >= secondsToSamps(cfg.minEmbedSec)) {
                    done = engine.enrollFromUtterance(seg);
                }
            }
            return done;
        }

        private void reset() {
            st = State.IDLE; preroll.clear(); voiced.clear(); silence = 0;
        }

        // small utils
        private static short[] padTo(short[] x, int n) {
            short[] y = new short[n];
            System.arraycopy(x, 0, y, 0, x.length);
            return y;
        }
        private int secondsToSamps(float s) { return (int)Math.round(s * cfg.rateHz); }

    }

    public OnboardingStream startOnboardingStream() { return new OnboardingStream(engine, cfg, vad); }

    // ---------- Onboarding (WAV) ----------
    public OnboardingResult onboardFromWav(File wav) throws Exception {
        short[] pcm = WavIO.readPcm16Mono16k(wav);
        return engine.enrollFromUtterance(pcm); // internally segments & builds cluster/mean
    }

    // ---------- Verification (MIC) ----------
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    public VerificationResult verifyFromMicrophone(long maxMillis) throws Exception {
        long deadline = System.currentTimeMillis() + Math.max(1_000, maxMillis);
        try (MicRecorder mic = new MicRecorder(appContext, cfg.rateHz, cfg.vadChunk)) {
            if (ContextCompat.checkSelfPermission(appContext, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            throw new SecurityException("RECORD_AUDIO permission not granted. " +
                    "Caller must request it before starting MicRecorder.");
            }
            mic.start();
            VerificationResult out;
            while (System.currentTimeMillis() < deadline) {
                short[] block = mic.readBlock();
                out = engine.pushVerify(block);
                if (out != null) return out;
            }
            // flush
            return engine.finishVerify();
        }
    }

    // ---------- Verification (STREAM) ----------
    public VerificationResult verifyStreamPush(short[] block) throws Exception {
        return engine.pushVerify(block);
    }
    public VerificationResult verifyStreamFinish() throws Exception {
        return engine.finishVerify();
    }

    // ---------- Verification (WAV) ----------
    public VerificationResult verifyFromWav(File wav) throws Exception {
        short[] pcm = WavIO.readPcm16Mono16k(wav);
        // feed in vadChunk-sized blocks
        int i = 0;
        VerificationResult r = null, last = null;
        while (i < pcm.length) {
            int take = Math.min(cfg.vadChunk, pcm.length - i);
            short[] b = java.util.Arrays.copyOfRange(pcm, i, i + take);
            i += take;
            if (b.length < cfg.vadChunk) {
                short[] pad = new short[cfg.vadChunk];
                System.arraycopy(b, 0, pad, 0, b.length);
                b = pad;
            }
            r = engine.pushVerify(b);
            if (r != null) last = r;
        }
        VerificationResult tail = engine.finishVerify();
        return (tail != null) ? tail : last;
    }

    // ---------- Export helpers ----------
    public Uri exportDefaultClusterToDownloads(Context ctx) throws IOException {
        return SpeakerIdStorage.exportToDownloads(ctx,
                SpeakerIdStorage.defaultClusterFile(ctx),
                "speaker_emb_cluster.npy");
    }
    public Uri exportDefaultMeanToDownloads(Context ctx) throws IOException {
        return SpeakerIdStorage.exportToDownloads(ctx,
                SpeakerIdStorage.defaultMeanEmbFile(ctx),
                "speaker_emb.npy");
    }
    public Uri exportDefaultMeanCountToDownloads(Context ctx) throws IOException {
        return SpeakerIdStorage.exportToDownloads(ctx,
                SpeakerIdStorage.defaultMeanCountFile(ctx),
                "speaker_emb.npy.count");
    }

    // mic onboarding helper (full-segment capture)
    private short[] collectFirstVoicedSegmentFromMic(MicRecorder mic, long deadline) {
        final int vadChunk = cfg.vadChunk;
        java.util.ArrayDeque<short[]> preroll = new java.util.ArrayDeque<>();
        int silence = 0; boolean active = false;
        ShortArray voiced = new ShortArray();

        while (System.currentTimeMillis() < deadline) {
            short[] block = mic.readBlock();
            float p = vad.feed(block);

            if (!active) {
                preroll.addLast(block);
                while (preroll.size() > cfg.prerollFrames) preroll.removeFirst();
                if (p >= cfg.onThr) {
                    active = true; for (short[] pr : preroll) voiced.append(pr);
                    voiced.append(block); silence = 0;
                }
            } else {
                if (p >= cfg.offThr) { voiced.append(block); silence = 0; }
                else {
                    silence += vadChunk;
                    if (silence >= (int)Math.round(cfg.silenceAfterSec * cfg.rateHz)) {
                        return voiced.toArray();
                    }
                }
            }
        }
        return null;
    }

    private int secondsToSamps(float s){ return (int)Math.round(s * cfg.rateHz); }

    // ==================== WWD: NEW PUBLIC APIS ====================

    /** Capture exactly 1.0 s of voiced audio (tight), else null on timeout. */
    private short[] collectExactly1sVoiced(MicRecorder mic, long deadlineMs) {
        final int target = secondsToSamps(1.0f);
        final int vadChunk = cfg.vadChunk;

        ShortArray out = new ShortArray();
        boolean active = false;
        int silence = 0;

        while (System.currentTimeMillis() < deadlineMs) {
            short[] b = mic.readBlock();
            if (b == null) continue;
            if (b.length < vadChunk) {
                short[] pad = new short[vadChunk];
                System.arraycopy(b, 0, pad, 0, b.length);
                b = pad;
            }
            float p = vad.feed(b);

            if (!active) {
                if (p >= cfg.onThr) {
                    active = true;
                    out.append(b);
                    silence = 0;
                }
            } else {
                if (p >= cfg.offThr) {
                    out.append(b);
                    silence = 0;
                    if (out.n >= target) {
                        // trim to exact 1s
                        short[] y = out.toArray();
                        return Arrays.copyOf(y, target);
                    }
                } else {
                    silence += vadChunk;
                    if (silence >= (int)Math.round(cfg.silenceAfterSec * cfg.rateHz)) {
                        // short speech; if at least 1s, trim; else keep waiting for a new activation
                        if (out.n >= target) {
                            short[] y = out.toArray();
                            return Arrays.copyOf(y, target);
                        }
                        // reset for next activation
                        out.clear(); active = false; silence = 0;
                    }
                }
            }
        }
        return null;
    }

    /**
     * Wake-word onboarding:
     * - captures 'embNum' separate 1.0 s voiced chunks
     * - embeds each (engine.embedOnce)
     * - enrolls cluster K=embNum and mean via engine.enrollFromEmbeddings(...)
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    public OnboardingResult onboardFromMicrophoneWWD(int embNum, long maxWallMs) throws Exception {
        if (embNum <= 0) embNum = 1;
        if (ContextCompat.checkSelfPermission(appContext, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            throw new SecurityException("RECORD_AUDIO permission not granted.");
        }
        final long deadline = System.currentTimeMillis() + Math.max(2_000, maxWallMs);

        // Ensure WWD slice settings are in effect (1.0s)
        Float prevHop = cfg.sliceHopSec;
        float prevSlice = cfg.sliceSec, prevMin = cfg.minEmbedSec;
        int prevK = cfg.clusterSize;
        try (MicRecorder mic = new MicRecorder(appContext, cfg.rateHz, cfg.vadChunk)) {
            cfg.sliceSec = 1.0f;
            cfg.sliceHopSec = 1.0f;
            cfg.minEmbedSec = 1.0f;
            cfg.clusterSize = embNum;

            mic.start();
            ArrayList<float[]> embs = new ArrayList<>(embNum);

            for (int i = 0; i < embNum; i++) {
                long perCapDeadline = deadline -  (embNum - 1 - i)* 1000L; // simple spreading
                short[] v1s = collectExactly1sVoiced(mic, perCapDeadline);
                if (v1s == null) {
                    throw new IllegalStateException("WWD onboard: failed to capture 1s voiced chunk #" + (i+1));
                }
                float[] e = engine.embedOnce(v1s);
                embs.add(e);
                Log.i(TAG, String.format(Locale.US, "[WWD] captured #%d embedding_len=%d", (i+1), e.length));
            }

            // Enroll directly from those embeddings
            return engine.enrollFromEmbeddings(embs);
        } finally {
            // restore
            cfg.sliceSec = prevSlice;
            cfg.sliceHopSec = prevHop;
            cfg.minEmbedSec = prevMin;
            cfg.clusterSize = prevK;
        }
    }

    /**
     * Wake-word verification:
     * - captures exactly 1.0 s of voiced audio
     * - scores only that chunk
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    public VerificationResult verifyFromMicrophoneWWD(long maxWallMs) throws Exception {
        if (ContextCompat.checkSelfPermission(appContext, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            throw new SecurityException("RECORD_AUDIO permission not granted.");
        }
        final long deadline = System.currentTimeMillis() + Math.max(2_000, maxWallMs);

        // Ensure WWD slice/min are active during this call (1.0 s)
        Float prevHop = cfg.sliceHopSec;
        float prevSlice = cfg.sliceSec, prevMin = cfg.minEmbedSec;

        try (MicRecorder mic = new MicRecorder(appContext, cfg.rateHz, cfg.vadChunk)) {
            cfg.sliceSec = 1.0f;
            cfg.sliceHopSec = 1.0f;
            cfg.minEmbedSec = 1.0f;

            mic.start();
            short[] v1s = collectExactly1sVoiced(mic, deadline);
            if (v1s == null) {
                throw new IllegalStateException("WWD verify: failed to capture 1.0 s voiced.");
            }
            return engine.verifyFromVoicedSegment(v1s);
        } finally {
            // restore
            cfg.sliceSec = prevSlice;
            cfg.sliceHopSec = prevHop;
            cfg.minEmbedSec = prevMin;
        }
    }

    /** Check whatever files the instance is currently configured to use (regular or WWD). */
    public boolean initVerificationUsingCurrentConfig() {
        try {
            return cfg != null
                    && cfg.meanEmbNpy != null && cfg.clusterNpy != null
                    && cfg.meanEmbNpy.exists() && cfg.clusterNpy.exists();
        } catch (Throwable t) {
            return false;
        }
    }

    // ==================== EXTERNAL-AUDIO CLUSTER API (KWD-style) ====================

    /** In-memory handle for a cluster managed via external raw audio buffers. */
    private static final class ExtCluster {
        final int id;
        final int capacity;                      // desired #embeddings to keep (FIFO)
        final ArrayDeque<float[]> fifo = new ArrayDeque<>();
        float[] mean;                            // L2-normalized running mean
        final File clusterFile;                  // KxD matrix for this cluster id
        final File meanFile;                     // 1xD vector for this cluster id

        ExtCluster(int id, int cap, File dir) {
            this.id = id;
            this.capacity = Math.max(1, cap);
            this.clusterFile = new File(dir, "spk_cluster_" + id + ".npy");
            this.meanFile    = new File(dir, "spk_mean_"    + id + ".npy");
        }
    }

    private final java.util.concurrent.atomic.AtomicInteger nextExtClusterId = new java.util.concurrent.atomic.AtomicInteger(1);
    private final Map<Integer, ExtCluster> extClusters = new HashMap<>();

    /** Create (or resume) a cluster that will hold up to numOfEmb embeddings, persisted per id. */
    public synchronized int initCluster(int numOfEmb) {
        int id = nextExtClusterId.getAndIncrement();
        ExtCluster cm = new ExtCluster(id, numOfEmb, appContext.getFilesDir());

        // If files exist, load them so sessions survive app restarts
        try {
            if (cm.clusterFile.exists()) {
                float[][] cl = NpyUtil.loadMatrixFloat32(cm.clusterFile);
                if (cl != null && cl.length > 0) {
                    for (float[] r : cl) {
                        l2normInPlaceLocal(r);
                        cm.fifo.addLast(r);
                    }
                }
            }
            if (cm.meanFile.exists()) {
                float[] m = null;
                try { m = NpyUtil.loadVectorFloat32(cm.meanFile); } catch (Throwable ignore) {}
                if (m == null) {
                    try {
                        float[][] mtx = NpyUtil.loadMatrixFloat32(cm.meanFile);
                        if (mtx != null && mtx.length == 1 && mtx[0] != null) m = java.util.Arrays.copyOf(mtx[0], mtx[0].length);
                    } catch (Throwable ignore) {}
                }
                if (m != null) l2normInPlaceLocal(m);
                cm.mean = m;
            }

            // If mean is missing but we have fifo, rebuild mean
            if (cm.mean == null && !cm.fifo.isEmpty()) {
                cm.mean = meanOfRowsLocal(cm.fifo.toArray(new float[0][]));
                l2normInPlaceLocal(cm.mean);
            }
        } catch (Throwable t) {
            // Non-fatal: start empty if load fails
            cm.fifo.clear();
            cm.mean = null;
        }

        extClusters.put(id, cm);
        return id;
    }

    // SpeakerIdApi.java

    /** Extract exactly the last 1.0 s of VAD-passing audio from pcm.
     * Policy:
     *  - Split into vadChunk frames
     *  - Keep ONLY frames where vad.feed(frame) >= 0.1
     *  - If total kept >= 1.0 s → return the LAST 1.0 s
     *  - If total kept == 0    → return 1.0 s of zeros
     *  - Else (0 < kept < 1.0 s) → DUPLICATE the kept sequence to reach 1.0 s
     */
    public short[] extractLast1sVoiced(short[] pcm) {
        final int rate = cfg.rateHz;              // e.g. 16000
        final int vadChunk = cfg.vadChunk;        // frame size used by VAD
        final int want = (int)Math.round(1.0f * rate);
        final float thr = 0.10f;                  // fixed threshold per your request

        if (pcm == null || pcm.length == 0) return new short[want];

        ShortArray voiced = new ShortArray();

        int i = 0;
        while (i < pcm.length) {
            int take = Math.min(vadChunk, pcm.length - i);
            short[] b = java.util.Arrays.copyOfRange(pcm, i, i + take);
            i += take;

            // pad to vadChunk before VAD
            if (b.length < vadChunk) {
                short[] pad = new short[vadChunk];
                System.arraycopy(b, 0, pad, 0, b.length);
                b = pad;
            }

            float p = vad.feed(b);
            if (p >= thr) {
                // keep ONLY voiced frames
                voiced.append(b);
            }
            // else: drop non-voiced completely
        }

        short[] allVoiced = voiced.toArray();

        if (allVoiced.length >= want) {
            // return the LAST 1.0 s
            short[] y = new short[want];
            System.arraycopy(allVoiced, allVoiced.length - want, y, 0, want);
            return y;
        }

        if (allVoiced.length == 0) {
            // nothing voiced → zeros
            return new short[want];
        }

        // DUPLICATE the voiced sequence to fill to 1.0 s
        short[] y = new short[want];
        int off = 0;
        while (off < want) {
            int copy = Math.min(allVoiced.length, want - off);
            System.arraycopy(allVoiced, 0, y, off, copy);
            off += copy;
        }
        return y;
    }

    /** Optional convenience if RN sends PCM16LE bytes. */
    public short[] extractLast1sVoiced(byte[] pcm16le) {
        final int want = (int)Math.round(1.0f * cfg.rateHz);
        if (pcm16le == null) return new short[want];
        int smp = pcm16le.length / 2;
        short[] x = new short[smp];
        for (int i=0, s=0; i+1<pcm16le.length; i+=2, s++) {
            int lo = (pcm16le[i] & 0xFF);
            int hi = (pcm16le[i+1] << 8);
            x[s] = (short)(hi | lo);
        }
        return extractLast1sVoiced(x);
    }

    private static final class ShortArray {
        short[] a = new short[0]; int n = 0;
        void append(short[] b){ ensure(n+b.length); System.arraycopy(b,0,a,n,b.length); n+=b.length; }
        short[] toArray(){ return java.util.Arrays.copyOf(a, n); }
        void clear(){ n = 0; }  // <-- added: used by voiced.clear() / out.clear()
        private void ensure(int m){ if(a.length>=m) return; a = java.util.Arrays.copyOf(a, Math.max(m, a.length*2+1024)); }
    }

    /**
     * Push exactly ONE embedding into the given cluster from a raw PCM buffer:
     * - use LAST 1.0 s (preferred),
     * - if shorter than 1.0 s, duplicate frames to pad up to exactly 1.0 s,
     * - FIFO if capacity exceeded,
     * - persist cluster + mean to disk every call.
     */
    public synchronized void createAndPushEmbeddingsToCluster(int clusterId, short[] pcm, int length) {
        ExtCluster cm = extClusters.get(clusterId);
        if (cm == null) throw new IllegalStateException("Unknown cluster id: " + clusterId);
        if (pcm == null) throw new IllegalArgumentException("pcm is null");
        length = Math.max(0, Math.min(length, pcm.length));

        try {
            short[] oneSec = buildOneSecondWindow(pcm, length);
            float[] emb = engine.embedOnce(oneSec); // returns L2-normalized
            // FIFO: keep at most capacity
            if (cm.fifo.size() >= cm.capacity) cm.fifo.removeFirst();
            cm.fifo.addLast(emb);

            // Update mean (simple average of unit vectors → re-normalize)
            cm.mean = meanOfRowsLocal(cm.fifo.toArray(new float[0][]));
            l2normInPlaceLocal(cm.mean);

            // Persist to disk
            persistExtCluster(cm);
        } catch (Throwable e) {
            // Spec says: never fail. As a last resort, embed 1s of zeros to keep consistency.
            try {
                short[] zeros = new short[secondsToSamps(1.0f)];
                float[] embZ = engine.embedOnce(zeros);
                if (cm.fifo.size() >= cm.capacity) cm.fifo.removeFirst();
                cm.fifo.addLast(embZ);
                cm.mean = meanOfRowsLocal(cm.fifo.toArray(new float[0][]));
                l2normInPlaceLocal(cm.mean);
                persistExtCluster(cm);
            } catch (Throwable ignore) {
                // If even zeros fail, swallow to obey "never fail" contract.
            }
        }
    }

    /**
     * Verify once against the given cluster from a raw PCM buffer:
     * - slice policy identical to push (LAST 1.0 s, pad by duplication),
     * - returns max cosine score vs {mean + all cluster rows},
     * - never throws.
     */
    public synchronized float createAndVerifyEmbeddingsFromCluster(int clusterId, short[] pcm, int length) {
        ExtCluster cm = extClusters.get(clusterId);
        if (cm == null) return Float.NEGATIVE_INFINITY;

        // Lazy load from disk if memory empty but files exist (e.g., app restarted)
        if (cm.fifo.isEmpty() && cm.clusterFile.exists()) {
            try {
                float[][] cl = NpyUtil.loadMatrixFloat32(cm.clusterFile);
                if (cl != null && cl.length > 0) {
                    for (float[] r : cl) { l2normInPlaceLocal(r); cm.fifo.addLast(r); }
                }
                if (cm.mean == null && cm.meanFile.exists()) {
                    float[] m = null;
                    try { m = NpyUtil.loadVectorFloat32(cm.meanFile); } catch (Throwable ignore) {}
                    if (m == null) {
                        try { float[][] mtx = NpyUtil.loadMatrixFloat32(cm.meanFile);
                            if (mtx != null && mtx.length == 1 && mtx[0] != null) m = java.util.Arrays.copyOf(mtx[0], mtx[0].length);
                        } catch (Throwable ignore) {}
                    }
                    if (m != null) { l2normInPlaceLocal(m); cm.mean = m; }
                }
            } catch (Throwable ignore) {}
        }
        if (cm.fifo.isEmpty() && cm.mean == null) return Float.NEGATIVE_INFINITY;

        try {
            length = Math.max(0, Math.min(length, pcm.length));
            short[] oneSec = buildOneSecondWindow(pcm, length);
            float[] q = engine.embedOnce(oneSec); // L2-normalized

            float best = (cm.mean != null) ? SpeakerEmbedderOrt.cosine(q, cm.mean) : Float.NEGATIVE_INFINITY;
            for (float[] r : cm.fifo) {
                float s = SpeakerEmbedderOrt.cosine(q, r);
                if (s > best) best = s;
            }
            return best; // "probability" as cosine score [-1,1]
        } catch (Throwable e) {
            // Spec: do not fail; return worst score if embedding fails for any reason.
            return Float.NEGATIVE_INFINITY;
        }
    }

    // ------------------------- helpers (local, no impact to engine) -------------------------

    /** Build exactly 1.0 s window: prefer LAST 1.0 s; if shorter, duplicate to fill. */
    private short[] buildOneSecondWindow(short[] pcm, int length) {
        int want = secondsToSamps(1.0f);
        if (length >= want) {
            int start = length - want;
            short[] y = new short[want];
            System.arraycopy(pcm, start, y, 0, want);
            return y;
        }
        // Duplicate (loop) to fill up to 1.0s
        if (length <= 0) return new short[want];
        short[] y = new short[want];
        int i = 0;
        while (i < want) {
            int take = Math.min(length, want - i);
            System.arraycopy(pcm, 0, y, i, take);
            i += take;
        }
        return y;
    }

    private void persistExtCluster(ExtCluster cm) throws IOException {
        // Save cluster (KxD)
        float[][] cl = cm.fifo.toArray(new float[0][]);
        try {
            NpyUtil.saveMatrixFloat32Atomic(cm.clusterFile, cl);
        } catch (NoSuchMethodError | UnsupportedOperationException ignore) {
            NpyUtil.saveMatrixFloat32(cm.clusterFile, cl);
        }
        // Save mean as 1-D vector for fast reload
        try {
            NpyUtil.saveVectorFloat32Atomic(cm.meanFile, cm.mean);
        } catch (NoSuchMethodError | UnsupportedOperationException ignore) {
            NpyUtil.saveVectorFloat32(cm.meanFile, cm.mean);
        }
    }

    private static float[] meanOfRowsLocal(float[][] m) {
        int rows = m.length, cols = m[0].length;
        float[] out = new float[cols];
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) out[c] += m[r][c];
        for (int c = 0; c < cols; ++c) out[c] /= Math.max(1, rows);
        return out;
    }
    private static void l2normInPlaceLocal(float[] v) {
        double ss = 1e-10;
        for (float x : v) ss += (double)x * x;
        float inv = (float)(1.0 / Math.sqrt(ss));
        for (int i = 0; i < v.length; ++i) v[i] *= inv;
    }

}
