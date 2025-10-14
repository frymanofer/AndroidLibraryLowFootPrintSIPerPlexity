package ai.perplexity.hotword.speakerid;

public final class OnboardingResult {
    public final int clusterSize;
    public final int embDim;
    public OnboardingResult(int clusterSize, int embDim) {
        this.clusterSize = clusterSize; this.embDim = embDim;
    }
}
