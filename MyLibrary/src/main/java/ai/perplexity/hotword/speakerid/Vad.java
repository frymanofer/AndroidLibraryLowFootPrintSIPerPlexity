package ai.perplexity.hotword.speakerid;

/** Minimal VAD interface expected by SpeakerIdEngine and VadAdapter. */
public interface Vad {
    /**
     * Feed one block of PCM16 at the detector sample rate; returns speech probability [0..1].
     * The block length should match your model’s frame size (e.g., Constants.FRAME_LENGTH).
     */
    float feed(short[] block16k);
}
