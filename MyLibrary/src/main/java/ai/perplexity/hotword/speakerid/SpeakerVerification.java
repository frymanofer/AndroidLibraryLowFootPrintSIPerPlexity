package ai.perplexity.hotword.speakerid;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

public final class SpeakerVerification {
    private static final int VERSION = 1;
    private static final int RATE_HZ = 16000;

    // Optional global embedder provider so createVerifierFromCluster(bytes) can be used
    private static volatile SpeakerIdApi defaultApi = null;

    /** Call once after you create the API (e.g., right after createWWD). */
    public static void setDefaultApi(SpeakerIdApi api) { defaultApi = api; }

    // -------- Factories --------

    /** Build from raw audio buffers (2s PCM16LE mono @16kHz each). */
    public static Verifier createVerifierFromAudio(SpeakerIdApi api, List<byte[]> buffers) throws Exception {
        Objects.requireNonNull(api, "SpeakerIdApi is required");
        if (buffers == null || buffers.isEmpty()) throw new IllegalArgumentException("buffers must not be empty");
        ArrayList<float[]> embs = new ArrayList<>(buffers.size());
        for (byte[] b : buffers) {
            short[] oneSec = last1sPcm16le(b);
            embs.add(api.embedOnce(oneSec)); // L2 unit
        }
        float[][] cluster = toMatrix(embs);
        float[] mean = meanOfRows(cluster); l2(mean);
        return new Verifier(api, cluster, mean);
    }

    /** Build from a cluster blob previously returned by Verifier.getCluster(). */
    public static Verifier createVerifierFromCluster(byte[] blob) throws IOException {
        Wrap w = decode(blob);
        // api is resolved later: per-instance if provided, else fallback to defaultApi
        return new Verifier(null, w.cluster, w.mean);
    }

    // -------- Verifier --------

    public static final class Verifier {
        private SpeakerIdApi api;          // optional; if null, uses defaultApi
        private final float[][] cluster;   // K x D (rows are unit)
        private final float[] mean;        // D (unit)

        private Verifier(SpeakerIdApi api, float[][] cluster, float[] mean) {
            this.api = api; this.cluster = cluster; this.mean = mean;
        }

        /** Optional: bind/override the API later (e.g., after restore). */
        public void bind(SpeakerIdApi api) { this.api = api; }

        /** Score new sample (2s PCM16LE mono @16kHz recommended; we use the LAST 1s). */
        public float verify(byte[] pcm16le) {
            try {
                SpeakerIdApi use = (api != null) ? api : defaultApi;
                if (use == null) throw new IllegalStateException("No embedder available. Call SpeakerVerification.setDefaultApi(...) or bind(...) first.");
                short[] oneSec = last1sPcm16le(pcm16le);
                float[] q = use.embedOnce(oneSec); // unit
                float best = cos(q, mean);
                for (float[] row : cluster) { float s = cos(q, row); if (s > best) best = s; }
                return best;
            } catch (Throwable t) {
                return Float.NEGATIVE_INFINITY;
            }
        }

        /** Serialize mean+cluster to one compact blob you can store anywhere. */
        public byte[] getCluster() { return encode(cluster, mean); }
        public int getK() { return cluster.length; }
        public int getD() { return cluster.length == 0 ? 0 : cluster[0].length; }
    }

    // -------- Binary format: "SPKC" | ver | K | D | mean[D] | cluster[K][D] (float32 LE) --------
    private static final class Wrap { float[][] cluster; float[] mean; }

    private static byte[] encode(float[][] cluster, float[] mean) {
        final int K = cluster.length, D = (K == 0 ? mean.length : cluster[0].length);
        ByteBuffer bb = ByteBuffer.allocate(4+4+4+4 + D*4 + K*D*4).order(ByteOrder.LITTLE_ENDIAN);
        bb.put((byte)'S').put((byte)'P').put((byte)'K').put((byte)'C'); bb.putInt(VERSION); bb.putInt(K); bb.putInt(D);
        for (int i=0;i<D;i++) bb.putFloat(mean[i]);
        for (int r=0;r<K;r++) for (int c=0;c<D;c++) bb.putFloat(cluster[r][c]);
        return bb.array();
    }
    private static Wrap decode(byte[] blob) throws IOException {
        try {
            ByteBuffer bb = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
            if (bb.remaining() < 16) throw new IOException("blob too short");
            if (bb.get()!='S' || bb.get()!='P' || bb.get()!='K' || bb.get()!='C') throw new IOException("bad magic");
            int ver = bb.getInt(); if (ver != VERSION) throw new IOException("unsupported version " + ver);
            int K = bb.getInt(), D = bb.getInt(); if (K<0 || D<=0) throw new IOException("bad shape");
            float[] mean = new float[D]; for (int i=0;i<D;i++) mean[i]=bb.getFloat(); l2(mean);
            float[][] cl = new float[K][D];
            for (int r=0;r<K;r++) for (int c=0;c<D;c++) cl[r][c]=bb.getFloat();
            for (float[] row : cl) l2(row);
            Wrap w = new Wrap(); w.cluster = cl; w.mean = mean; return w;
        } catch (RuntimeException e) { throw new IOException("decode failed: " + e.getMessage(), e); }
    }

    // -------- Audio + math helpers --------
    private static short[] last1sPcm16le(byte[] le16) {
        if (le16 == null) return new short[RATE_HZ];
        int smp = le16.length / 2;
        short[] pcm = new short[smp];
        for (int i=0,s=0;i+1<le16.length;i+=2,s++){ int lo=(le16[i]&0xFF); int hi=(le16[i+1]<<8); pcm[s]=(short)(hi|lo); }
        return last1sPcm16(pcm);
    }
    private static short[] last1sPcm16(short[] pcm) {
        int want = RATE_HZ;
        if (pcm.length >= want) { short[] y=new short[want]; System.arraycopy(pcm, pcm.length-want, y, 0, want); return y; }
        if (pcm.length <= 0) return new short[want];
        short[] y=new short[want]; int i=0; while(i<want){ int take=Math.min(pcm.length, want-i); System.arraycopy(pcm,0,y,i,take); i+=take; } return y;
    }
    private static float[][] toMatrix(List<float[]> rows){ int K=rows.size(),D=rows.get(0).length; float[][] m=new float[K][D]; for(int r=0;r<K;r++) System.arraycopy(rows.get(r),0,m[r],0,D); return m; }
    private static float[] meanOfRows(float[][] m){ int K=m.length,D=m[0].length; float[] out=new float[D]; for(int r=0;r<K;r++) for(int c=0;c<D;c++) out[c]+=m[r][c]; for(int c=0;c<D;c++) out[c]/=Math.max(1,K); return out; }
    private static void l2(float[] v){ double ss=1e-10; for(float x:v) ss+=x*x; float inv=(float)(1.0/Math.sqrt(ss)); for(int i=0;i<v.length;i++) v[i]*=inv; }
    private static float cos(float[] a,float[] b){ float s=0f; for(int i=0;i<a.length;i++) s+=a[i]*b[i]; return s; }
}
