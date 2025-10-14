package ai.perplexity.hotword.speakerid;

import java.util.Map;

public final class VerificationResult {
    public final float fullSec;
    public final float voicedSec;
    public final float bestScore;
    public final String bestStrategy;
    public final String bestTargetLabel;
    public final Map<String, Map<String, Float>> perTargetStrategy;

    public VerificationResult(float fullSec, float voicedSec, float bestScore,
                              String bestStrategy, String bestTargetLabel,
                              Map<String, Map<String, Float>> perTargetStrategy) {
        this.fullSec = fullSec; this.voicedSec = voicedSec; this.bestScore = bestScore;
        this.bestStrategy = bestStrategy; this.bestTargetLabel = bestTargetLabel;
        this.perTargetStrategy = perTargetStrategy;
    }
}
