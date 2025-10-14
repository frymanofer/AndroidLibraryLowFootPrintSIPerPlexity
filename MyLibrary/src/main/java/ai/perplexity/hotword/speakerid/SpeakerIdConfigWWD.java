package ai.perplexity.hotword.speakerid;

import java.io.File;
import java.util.Arrays;

public final class SpeakerIdConfigWWD {
    public int   rateHz            = 16000;
    public int   vadChunk          = 1280;  // 80ms at 16k
    public float onThr             = 0.45f;
    public float offThr            = 0.30f;
    public float silenceAfterSec   = 2.0f;
    public int   prerollFrames     = 2;
    // SpeakerIdConfig.java  (add fields with sensible defaults)
    public float  onboardVoicedTargetSec = 3.0f; // need this much voiced audio in ONE segment
    public boolean debugVadFrames = true;       // spammy logs per VAD frame

    public float sliceSec          = 1.0f;
    public Float sliceHopSec       = 1.0f;
    public float minEmbedSec       = 1.0f;

    public boolean flexEnabled     = false;
    public float[] flexSizesSec    = new float[]{0.25f, 0.50f, 0.75f, 1.00f};
    public float flexMaxSec        = 1.50f;
    public int   flexTopK          = 3;

    public int   clusterSize       = 5;

    /** Online adaptation (disabled if <0). */
    public float addSampleThreshold = -1f;
    public int   addSampleMax       = 1_000_000;

    /** Paths where we persist mean vector and cluster (NumPy .npy format). */
    public File  meanEmbNpy;      // "speaker_emb.npy"
    public File  clusterNpy;      // "speaker_emb_cluster.npy"

    public SpeakerIdConfig copy() {
        SpeakerIdConfig c = new SpeakerIdConfig();
        c.rateHz = rateHz; c.vadChunk = vadChunk; c.onThr = onThr; c.offThr = offThr;
        c.silenceAfterSec = silenceAfterSec; c.prerollFrames = prerollFrames;
        c.sliceSec = sliceSec; c.sliceHopSec = sliceHopSec; c.minEmbedSec = minEmbedSec;
        c.flexEnabled = flexEnabled; c.flexSizesSec = Arrays.copyOf(flexSizesSec, flexSizesSec.length);
        c.flexMaxSec = flexMaxSec; c.flexTopK = flexTopK;
        c.clusterSize = clusterSize; c.addSampleThreshold = addSampleThreshold; c.addSampleMax = addSampleMax;
        c.meanEmbNpy = meanEmbNpy; c.clusterNpy = clusterNpy;
        return c;
    }
}
