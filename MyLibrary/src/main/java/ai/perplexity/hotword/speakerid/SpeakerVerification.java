package ai.perplexity.hotword.speakerid;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

public final class SpeakerVerification {
    private static final int VERSION = 2; // v2: cluster-only (no mean in blob)
    private static final int RATE_HZ = 16000;

    // Optional global embedder provider so createVerifierFromCluster(bytes) can be used
    private static volatile SpeakerIdApi defaultApi = null;

    /** Call once after you create the API (e.g., right after createWWD). */
    public static void setDefaultApi(SpeakerIdApi api) { defaultApi = api; }

    // -------- Factories --------
/** Duplicate-pad x until it reaches want samples (deterministic). */
private static short[] padByDup(short[] x, int want) {
    if (x == null || x.length == 0) return new short[want];
    if (x.length >= want) return java.util.Arrays.copyOf(x, want);
    short[] y = new short[want];
    int k = 0;
    while (k < want) {
        int take = Math.min(x.length, want - k);
        System.arraycopy(x, 0, y, k, take);
        k += take;
    }
    return y;
}

public static Verifier createVerifierFromAudio(SpeakerIdApi api, List<byte[]> buffers) throws Exception {
    Objects.requireNonNull(api, "SpeakerIdApi is required");
    if (buffers == null || buffers.isEmpty())
        throw new IllegalArgumentException("buffers must not be empty");

    final int rate = RATE_HZ;
    final int win = rate;   // 1.0s
    final int hop = rate;   // 1.0s

    ArrayList<float[]> embs = new ArrayList<>();

    for (byte[] b : buffers) {
        // Tail-first, VAD-only voiced from last tailSec (matches Python --tail-sec path in SpeakerIdApi)
        short[] voiced = api.extractLast1sVoiced(b);

        // Base 1.0s slices (win=hop=1.0s)
        for (int i = 0; i + win <= voiced.length; i += hop) {
            short[] sl = java.util.Arrays.copyOfRange(voiced, i, i + win);
            embs.add(api.embedOnce(sl)); // already L2-normalized
        }

        // Tail remainder → pad by duplication to 1.0s and embed
        int rem = voiced.length % hop;
        if (rem > 0) {
            short[] tail = java.util.Arrays.copyOfRange(voiced, voiced.length - rem, voiced.length);
            short[] pad1s = padByDup(tail, win);
            embs.add(api.embedOnce(pad1s));
        }
    }

    // Make the cluster from all rows
    float[][] cluster = toMatrix(embs);

    // Append L2-mean row as last row (Python targets are {mean + rows})
    float[] mean = meanOfRows(cluster);
    l2(mean);
    embs.add(mean);
    cluster = toMatrix(embs);

    return new Verifier(api, cluster);
}


    /** Build from a cluster blob previously returned by Verifier.getCluster() (v2: cluster-only). */
    public static Verifier createVerifierFromCluster(byte[] blob) throws IOException {
        float[][] cluster = decodeClusterOnly(blob);
        return new Verifier(null, cluster);
    }

    // -------- Verifier --------

    public static final class Verifier {
        private SpeakerIdApi api;          // optional; if null, uses defaultApi
        private final float[][] cluster;   // K x D (rows are L2-unit). Last row may be mean.

        private Verifier(SpeakerIdApi api, float[][] cluster) {
            this.api = api;
            this.cluster = cluster;
        }

        /** Optional: bind/override the API later (e.g., after restore). */
        public void bind(SpeakerIdApi api) { this.api = api; }

/** Cap to capSec seconds and ensure at least minSec seconds by duplication padding. */
private static short[] capFloorWhole(short[] x, float capSec, float minSec, int rate) {
    if (x == null) x = new short[0];
    int cap = Math.max(1, Math.round(capSec * rate));
    int min = Math.max(1, Math.round(minSec * rate));
    short[] y = java.util.Arrays.copyOf(x, Math.min(x.length, cap));
    return (y.length >= min) ? y : padByDup(y, min);
}

        public float verify(byte[] pcm16le) {
            try {
                SpeakerIdApi use = (api != null) ? api : defaultApi;
                if (use == null) throw new IllegalStateException("No embedder available.");

                final int rate = RATE_HZ;
                final int win = rate;   // 1.0s
                final int hop = rate;   // 1.0s

                // Tail-first, VAD-only voiced from last tailSec (matches Python)
                short[] voiced = use.extractLast1sVoiced(pcm16le);

                // Build query set (FLEX-lite): 1.0s base slices + tail remainder padded to 1.0s + "whole" capped to 1.5s, floored to 0.25s
                ArrayList<float[]> queries = new ArrayList<>();

                // 1.0s slices
                for (int i = 0; i + win <= voiced.length; i += hop) {
                    short[] sl = java.util.Arrays.copyOfRange(voiced, i, i + win);
                    queries.add(use.embedOnce(sl)); // L2 unit
                }

                // remainder padded to 1.0s
                int rem = voiced.length % hop;
                if (rem > 0) {
                    short[] tail = java.util.Arrays.copyOfRange(voiced, voiced.length - rem, voiced.length);
                    short[] pad1s = padByDup(tail, win);
                    queries.add(use.embedOnce(pad1s));
                }

                // "whole" strategy: cap to 1.5s, floor at 0.25s (pad by duplication)
                short[] whole = capFloorWhole(voiced, /*capSec=*/1.5f, /*minSec=*/0.25f, rate);
                queries.add(use.embedOnce(whole));

                // Best-of scoring: max over (queries x cluster rows)
                float best = Float.NEGATIVE_INFINITY;
                for (float[] q : queries) {
                    for (float[] row : cluster) {
                        float s = cos(q, row);
                        if (s > best) best = s;
                    }
                }
                return best;
            } catch (Throwable t) {
                return Float.NEGATIVE_INFINITY;
            }
        }

        /** Return EXACTLY the cluster we hold (v2 format). */
        public byte[] getCluster() {
            return encodeClusterOnly(cluster);
        }

        public int getK() { return cluster.length; }
        public int getD() { return cluster.length == 0 ? 0 : cluster[0].length; }
    }

    // -------- Binary format v2: "SPKC" | ver(=2) | K | D | cluster[K][D] (float32 LE) --------

    private static byte[] encodeClusterOnly(float[][] cluster) {
        final int K = (cluster == null ? 0 : cluster.length);
        final int D = (K == 0 ? 0 : cluster[0].length);
        ByteBuffer bb = ByteBuffer.allocate(4 + 4 + 4 + 4 + K * D * 4).order(ByteOrder.LITTLE_ENDIAN);
        bb.put((byte) 'S').put((byte) 'P').put((byte) 'K').put((byte) 'C');
        bb.putInt(VERSION);
        bb.putInt(K);
        bb.putInt(D);
        for (int r = 0; r < K; r++) for (int c = 0; c < D; c++) bb.putFloat(cluster[r][c]);
        return bb.array();
    }

    private static float[][] decodeClusterOnly(byte[] blob) throws IOException {
        try {
            ByteBuffer bb = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
            if (bb.remaining() < 16) throw new IOException("blob too short");
            if (bb.get() != 'S' || bb.get() != 'P' || bb.get() != 'K' || bb.get() != 'C')
                throw new IOException("bad magic");
            int ver = bb.getInt();
            if (ver != VERSION) throw new IOException("unsupported version " + ver + " (expected " + VERSION + ")");
            int K = bb.getInt(), D = bb.getInt();
            if (K < 0 || D <= 0) throw new IOException("bad shape");
            if (bb.remaining() < (long) K * D * 4) throw new IOException("blob truncated");

            float[][] cl = new float[K][D];
            for (int r = 0; r < K; r++) for (int c = 0; c < D; c++) cl[r][c] = bb.getFloat();
            for (float[] row : cl) l2(row); // ensure rows are unit
            return cl;
        } catch (RuntimeException e) {
            throw new IOException("decode failed: " + e.getMessage(), e);
        }
    }

    // -------- Audio + math helpers --------

    /** Force to EXACTLY 1.0s @16k by duplication or trimming (deterministic). */
    private static short[] toExactly1s(short[] x) {
        final int want = RATE_HZ;
        if (x == null || x.length == 0) return new short[want];
        if (x.length == want) return x;
        if (x.length > want) return Arrays.copyOfRange(x, x.length - want, x.length); // keep last 1s
        short[] y = new short[want];
        int i = 0;
        while (i < want) {
            int take = Math.min(x.length, want - i);
            System.arraycopy(x, 0, y, i, take);
            i += take;
        }
        return y;
    }

    private static float[][] toMatrix(List<float[]> rows) {
        int K = rows.size(), D = rows.get(0).length;
        float[][] m = new float[K][D];
        for (int r = 0; r < K; r++) System.arraycopy(rows.get(r), 0, m[r], 0, D);
        return m;
    }

    private static float[] meanOfRows(float[][] m) {
        int K = m.length, D = m[0].length;
        float[] out = new float[D];
        for (int r = 0; r < K; r++) for (int c = 0; c < D; c++) out[c] += m[r][c];
        for (int c = 0; c < D; c++) out[c] /= Math.max(1, K);
        return out;
    }

    private static void l2(float[] v) {
        double ss = 1e-10;
        for (float x : v) ss += x * x;
        float inv = (float) (1.0 / Math.sqrt(ss));
        for (int i = 0; i < v.length; i++) v[i] *= inv;
    }

    private static float cos(float[] a, float[] b) {
        float s = 0f;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }
}
