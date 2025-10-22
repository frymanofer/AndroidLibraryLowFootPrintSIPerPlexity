package ai.perplexity.hotword.speakerid;

import java.io.File;
import java.util.Arrays;

public final class SpeakerIdConfigWWD {
    public int   rateHz            = 16000;
    public int   vadChunk          = 1280;     // 80ms @ 16k  (python --vad-chunk 1280)
    public float onThr             = 0.50f;    // python --on 0.5
    public float offThr            = onThr / 10.0f;    // used only by state-machine segmentation
    public float silenceAfterSec   = 2.0f;     // used only by state-machine segmentation
    public int   prerollFrames     = 0;        // strict parity: do not include pre-activation frames

    /** Use ONLY the last tailSec seconds BEFORE VAD selection (python --tail-sec). */
    public float tailSec           = 1.5f;     // python --tail-sec 1.5

    // This is only used by the state-machine onboarding path (mic); the collect-voiced path ignores it.
    public float  onboardVoicedTargetSec = 3.0f;
    public boolean debugVadFrames = true;

    // Slice/min settings used by some engine strategies; python default min_embed_sec=0.25
    public float sliceSec          = 1.0f;
    public Float sliceHopSec       = 1.0f;
    public float minEmbedSec       = 0.25f;    // python default

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

        c.tailSec = tailSec;                 // <— add this to SpeakerIdConfig

        c.sliceSec = sliceSec; c.sliceHopSec = sliceHopSec; c.minEmbedSec = minEmbedSec;
        c.flexEnabled = flexEnabled; c.flexSizesSec = Arrays.copyOf(flexSizesSec, flexSizesSec.length);
        c.flexMaxSec = flexMaxSec; c.flexTopK = flexTopK;
        c.clusterSize = clusterSize; c.addSampleThreshold = addSampleThreshold; c.addSampleMax = addSampleMax;
        c.meanEmbNpy = meanEmbNpy; c.clusterNpy = clusterNpy;
        return c;
    }
}
