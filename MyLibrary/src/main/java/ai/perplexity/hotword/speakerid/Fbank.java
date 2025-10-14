// MyLibrary/src/main/java/com/davoice/speakerid/Fbank.java
package ai.perplexity.hotword.speakerid;

import java.util.Arrays;

final class Fbank {
    static final class Config {
        final int sampleRate;     // 16000
        final int frameLen;       // 25 ms → 400 samples
        final int frameShift;     // 10 ms → 160 samples
        final int nFft;           // 512
        final int nMels;          // 80
        final float fMin;         // 20 Hz
        final float fMax;         // 7600 Hz (or sampleRate/2 - 400)
        final boolean useLog;     // true
        final boolean snipEdges;  // true (Kaldi/Sherpa default)
        final float dither;       // 0.0 (deterministic)

        Config(int sr) {
            this.sampleRate = sr;
            this.frameLen   = Math.round(0.025f * sr); // 25 ms
            this.frameShift = Math.round(0.010f * sr); // 10 ms
            this.nFft       = 512;
            this.nMels      = 80;
            this.fMin       = 20f;
            this.fMax       = Math.min(7600f, sr * 0.5f - 400f);
            this.useLog     = true;
            this.snipEdges  = true;
            this.dither     = 0f;
        }
    }

    private final Config cfg;
    private final float[] hann;           // window[frameLen]
    private final float[][] melFilters;   // [nMels][nFft/2+1]
    private final float logFloor = (float)Math.log(1e-10);

    Fbank(Config c) {
        this.cfg = c;
        this.hann = makeHann(c.frameLen);
        this.melFilters = buildMelTriFilters(c);
    }

    /** Main entry: short[] pcm16 → [T, nMels] features (log Mel), with per-bin mean norm (CMN). */
    float[][] compute(short[] pcm) {
        if (pcm == null || pcm.length == 0) return new float[0][0];

        // Convert to float -1..1
        float[] x = new float[pcm.length];
        for (int i = 0; i < pcm.length; i++) x[i] = pcm[i] / 32768f;

        // Dither (optional, default 0)
        if (cfg.dither > 0f) {
            for (int i = 0; i < x.length; i++) x[i] += cfg.dither * (float)gauss();
        }

        // Framing
        int T;
        if (cfg.snipEdges) {
            if (x.length < cfg.frameLen) return new float[0][0];
            T = 1 + (x.length - cfg.frameLen) / cfg.frameShift;
        } else {
            T = (int)Math.ceil((x.length - cfg.frameLen) / (double)cfg.frameShift) + 1;
        }
        float[][] feats = new float[T][cfg.nMels];

        float[] re = new float[cfg.nFft];
        float[] im = new float[cfg.nFft];
        float[] pow = new float[cfg.nFft/2 + 1];

        for (int t = 0; t < T; t++) {
            int start = t * cfg.frameShift;
            // center frame if !snipEdges; we keep snipEdges=true → simple start index
            // Copy frame with window
            Arrays.fill(re, 0f);
            Arrays.fill(im, 0f);

            for (int i = 0; i < cfg.frameLen; i++) {
                int idx = start + i;
                float s = 0f;
                if (idx >= 0 && idx < x.length) s = x[idx];
                re[i] = s * hann[i];
            }

            // FFT (real → complex)
            fftRadix2(re, im); // in-place

            // Power spectrum
            for (int k = 0; k <= cfg.nFft/2; k++) {
                float rr = re[k], ii = im[k];
                pow[k] = rr*rr + ii*ii;
            }

            // Apply Mel filters
            float[] m = feats[t];
            Arrays.fill(m, 0f);
            for (int mIx = 0; mIx < cfg.nMels; mIx++) {
                float e = 0f;
                float[] w = melFilters[mIx];
                for (int k = 0; k < w.length; k++) {
                    e += w[k] * pow[k];
                }
                if (cfg.useLog) {
                    m[mIx] = (e > 1e-10f) ? (float)Math.log(e) : logFloor;
                } else {
                    m[mIx] = e;
                }
            }
        }

        // Per-bin CMN (subtract mean over time), like common speaker pipelines
        cmnInPlace(feats);

        return feats;
    }

    // ---- helpers ----

    private static float[] makeHann(int n) {
        float[] w = new float[n];
        for (int i = 0; i < n; i++) {
            w[i] = 0.5f - 0.5f * (float)Math.cos(2.0 * Math.PI * i / Math.max(1, n-1));
        }
        return w;
    }

    private static double hzToMel(double f) {
        // Slaney mel (HTK alternative is fine too; Slaney is what Kaldi/kaldifeat uses)
        return 2595.0 * Math.log10(1.0 + f / 700.0);
    }
    private static double melToHz(double m) {
        return 700.0 * (Math.pow(10.0, m / 2595.0) - 1.0);
    }

    private static int freqToBin(double f, int nFft, int sr) {
        int bin = (int)Math.round(f * (nFft / 2.0) / (sr / 2.0));
        return Math.max(0, Math.min(nFft/2, bin));
    }

    private static float[][] buildMelTriFilters(Config c) {
        int nMels = c.nMels;
        int nBins = c.nFft/2 + 1;

        double melMin = hzToMel(c.fMin);
        double melMax = hzToMel(c.fMax);
        double melStep = (melMax - melMin) / (nMels + 1);

        int[] edges = new int[nMels + 2];
        for (int m = 0; m < edges.length; m++) {
            double mel = melMin + m * melStep;
            double hz  = melToHz(mel);
            edges[m]   = freqToBin(hz, c.nFft, c.sampleRate);
        }

        float[][] out = new float[nMels][nBins];
        for (int m = 1; m <= nMels; m++) {
            int left = edges[m - 1];
            int center = edges[m];
            int right = edges[m + 1];

            for (int k = left; k <= center; k++) {
                if (k >= 0 && k < nBins && center > left) {
                    out[m-1][k] = (k - left) / (float)(center - left);
                }
            }
            for (int k = center; k <= right; k++) {
                if (k >= 0 && k < nBins && right > center) {
                    out[m-1][k] = (right - k) / (float)(right - center);
                }
            }
        }
        return out;
    }

    private static void cmnInPlace(float[][] x) {
        if (x.length == 0) return;
        int T = x.length, D = x[0].length;
        float[] mean = new float[D];
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++)
                mean[d] += x[t][d];
        for (int d = 0; d < D; d++) mean[d] /= Math.max(1, T);
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++)
                x[t][d] -= mean[d];
    }

    // Simple in-place radix-2 FFT for real input in re[], imag in im[]
    private static void fftRadix2(float[] re, float[] im) {
        final int n = re.length;
        // bit-reverse
        int j = 0;
        for (int i = 1; i < n; i++) {
            int bit = n >>> 1;
            for (; (j & bit) != 0; bit >>>= 1) j &= ~bit;
            j |= bit;
            if (i < j) {
                float tr = re[i]; re[i] = re[j]; re[j] = tr;
                float ti = im[i]; im[i] = im[j]; im[j] = ti;
            }
        }
        // Cooley–Tukey
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2.0 * Math.PI / len;
            float wlenRe = (float)Math.cos(ang);
            float wlenIm = (float)Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                float wr = 1f, wi = 0f;
                for (int k = 0; k < len/2; k++) {
                    int u = i + k;
                    int v = u + len/2;
                    float vr = re[v] * wr - im[v] * wi;
                    float vi = re[v] * wi + im[v] * wr;
                    re[v] = re[u] - vr;
                    im[v] = im[u] - vi;
                    re[u] += vr;
                    im[u] += vi;
                    // w *= wlen
                    float tmp = wr * wlenRe - wi * wlenIm;
                    wi = wr * wlenIm + wi * wlenRe;
                    wr = tmp;
                }
            }
        }
    }

    // tiny Gaussian (Box–Muller); used only if dither>0
    private static double gauss() {
        double u = Math.random();
        double v = Math.random();
        return Math.sqrt(-2.0*Math.log(u+1e-12)) * Math.cos(2*Math.PI*v);
    }
}
