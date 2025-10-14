package ai.perplexity.hotword.speakerid;

import android.util.Log;

import java.io.*;
import java.util.*;

// --- tiny helper for growing short buffers ---
class ShortArray {
    short[] a = new short[0];
    int n = 0;
    void clear() { n = 0; }
    void append(short[] b) {
        if (b == null || b.length == 0) return;
        ensure(n + b.length);
        System.arraycopy(b, 0, a, n, b.length);
        n += b.length;
    }
    short[] toArray() { return Arrays.copyOf(a, n); }
    int size() { return n; }
    private void ensure(int m) {
        if (a.length >= m) return;
        int cap = Math.max(m, Math.max(1024, a.length * 2));
        a = Arrays.copyOf(a, cap);
    }
}
