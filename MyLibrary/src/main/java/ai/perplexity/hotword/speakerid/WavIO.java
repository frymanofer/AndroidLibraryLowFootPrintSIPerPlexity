package ai.perplexity.hotword.speakerid;

import java.io.*;

final class WavIO {
    private WavIO() {}

    /** Read PCM16 mono 16kHz WAV into a short[]. Throws if format mismatches. */
    static short[] readPcm16Mono16k(File f) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(f)))) {
            byte[] riff = new byte[4]; in.readFully(riff);
            if (!new String(riff, "US-ASCII").equals("RIFF")) throw new IOException("Not RIFF");
            in.skipBytes(4); // file size
            byte[] wave = new byte[4]; in.readFully(wave);
            if (!new String(wave, "US-ASCII").equals("WAVE")) throw new IOException("Not WAVE");

            int audioFormat = -1, numCh = -1, sampleRate = -1, bitsPerSample = -1;
            int dataLen = -1;

            while (true) {
                byte[] chunkId = new byte[4];
                if (in.read(chunkId) != 4) break;
                int chunkSize = Integer.reverseBytes(in.readInt());
                String id = new String(chunkId, "US-ASCII");
                if ("fmt ".equals(id)) {
                    audioFormat = Short.reverseBytes(in.readShort()) & 0xFFFF;
                    numCh       = Short.reverseBytes(in.readShort()) & 0xFFFF;
                    sampleRate  = Integer.reverseBytes(in.readInt());
                    in.skipBytes(6); // byte rate (4), block align (2)
                    bitsPerSample = Short.reverseBytes(in.readShort()) & 0xFFFF;
                    int remain = chunkSize - 16;
                    if (remain > 0) in.skipBytes(remain);
                } else if ("data".equals(id)) {
                    dataLen = chunkSize;
                    break;
                } else {
                    in.skipBytes(chunkSize);
                }
            }

            if (audioFormat != 1) throw new IOException("Only PCM supported");
            if (numCh != 1) throw new IOException("Only mono supported");
            if (sampleRate != 16000) throw new IOException("Only 16kHz supported");
            if (bitsPerSample != 16) throw new IOException("Only 16-bit supported");
            if (dataLen < 0) throw new IOException("No data chunk");

            int samples = dataLen / 2;
            short[] out = new short[samples];
            for (int i = 0; i < samples; ++i) out[i] = Short.reverseBytes(in.readShort());
            return out;
        }
    }
}
