package ai.perplexity.hotword.speakerid;

import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.*;

/**
 * DM extractor using the same delimiter + TAR split approach as KeyWordsDetection.
 */
final class DmExtractor {
    private static final String TAG = "DmExtractor";
    // MUST match your KeyWordsDetection DM delimiter exactly
    private static final String DM_DELIM = "####%$#$%$#^&^*&*^#$%#$%#$#%^&&*****###";

    /** Expand a .dm into outDir and return absolute paths for exact file names found. */
    static Map<String, String> expandDmAndPick(String dmPath, File outDir, String... exactNames) {
        Map<String, String> out = new HashMap<>();
        try {
            if (dmPath == null) return out;
            File dmFile = new File(dmPath);
            if (!dmFile.exists()) return out;

            if (!outDir.exists() && !outDir.mkdirs()) {
                Log.w(TAG, "expandDmAndPick: could not create " + outDir);
            }

            byte[] all = readAllBytes(dmPath);
            byte[] delim = DM_DELIM.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            List<int[]> ranges = splitByDelimiter(all, delim);

            for (int[] r : ranges) extractTarFromBuffer(all, r[0], r[1], outDir);

            // Exact filename matches
            for (String name : exactNames) {
                File f = findFileByExactName(outDir, name);
                if (f != null && f.exists()) {
                    out.put(name, f.getAbsolutePath());
                    Log.d(TAG, "expandDmAndPick: found exact " + name + " -> " + f.getAbsolutePath());
                } else {
                    Log.d(TAG, "expandDmAndPick: MISSING exact " + name);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "expandDmAndPick failed: " + e);
        }
        return out;
    }

    /** Heuristic: search for the first .onnx whose name contains any of the hints (case-insensitive). */
    static File findFirstByHeuristic(File dir, String... nameHints) {
        if (dir == null || !dir.exists()) return null;
        Queue<File> q = new ArrayDeque<>();
        q.add(dir);
        while (!q.isEmpty()) {
            File cur = q.remove();
            File[] kids = cur.listFiles();
            if (kids == null) continue;
            for (File f : kids) {
                if (f.isDirectory()) { q.add(f); continue; }
                String n = f.getName().toLowerCase(Locale.ROOT);
                if (!n.endsWith(".onnx")) continue;
                for (String h : nameHints) {
                    if (n.contains(h.toLowerCase(Locale.ROOT))) return f;
                }
            }
        }
        return null;
    }

    // ---------- helpers (same as your KD logic) ----------

    private static byte[] readAllBytes(String path) throws Exception {
        try (BufferedInputStream in = new BufferedInputStream(new FileInputStream(path))) {
            byte[] buf = new byte[8192];
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            int n;
            while ((n = in.read(buf)) != -1) baos.write(buf, 0, n);
            return baos.toByteArray();
        }
    }

    private static List<int[]> splitByDelimiter(byte[] data, byte[] delim) {
        List<int[]> ranges = new ArrayList<>();
        int start = 0, i = 0;
        while (i <= data.length - delim.length) {
            boolean match = true;
            for (int j = 0; j < delim.length; j++) {
                if (data[i + j] != delim[j]) { match = false; break; }
            }
            if (match) {
                ranges.add(new int[]{start, i});
                i += delim.length;
                start = i;
            } else {
                i++;
            }
        }
        if (start < data.length) ranges.add(new int[]{start, data.length});
        return ranges;
    }

    /** Minimal TAR extractor (ustar, store-only). */
    private static void extractTarFromBuffer(byte[] data, int start, int end, File outDir) throws Exception {
        int pos = start;
        while (pos + 512 <= end) {
            boolean allZero = true;
            for (int k = 0; k < 512; k++) {
                if (data[pos + k] != 0) { allZero = false; break; }
            }
            if (allZero) break;

            String name   = readNullTermString(data, pos + 0,   100);
            long size     = parseOctal(data,       pos + 124,   12);
            int typeflag  = (data[pos + 156] == 0) ? '0' : (data[pos + 156] & 0xFF);
            String prefix = readNullTermString(data, pos + 345, 155);
            String fullName = (prefix != null && !prefix.isEmpty()) ? (prefix + "/" + name) : name;

            pos += 512; // now at file payload

            if (typeflag == '5') {
                File dir = new File(outDir, fullName);
                if (!dir.exists()) dir.mkdirs();
            } else if (typeflag == '0' || typeflag == 0) {
                File out = new File(outDir, fullName);
                File parent = out.getParentFile();
                if (parent != null && !parent.exists()) parent.mkdirs();
                try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(out))) {
                    long remaining = size;
                    while (remaining > 0) {
                        int chunk = (int)Math.min(8192, remaining);
                        bos.write(data, pos, chunk);
                        pos += chunk;
                        remaining -= chunk;
                    }
                }
                int pad = (int)((512 - (size % 512)) % 512);
                pos += pad;
                continue;
            } else {
                int pad = (int)((512 - (size % 512)) % 512);
                pos += size + pad;
            }
        }
    }

    private static String readNullTermString(byte[] b, int off, int len) {
        int end = off + len, i = off;
        while (i < end && b[i] != 0) i++;
        return new String(b, off, i - off, java.nio.charset.StandardCharsets.US_ASCII).trim();
    }

    private static long parseOctal(byte[] b, int off, int len) {
        long val = 0; int end = off + len, i = off;
        while (i < end && b[i] == 0x20) i++; // skip spaces
        for (; i < end; i++) {
            byte c = b[i];
            if (c == 0 || c == 0x20) break;
            val = (val << 3) + (c - '0');
        }
        return val;
    }

    private static File findFileByExactName(File dir, String exact) {
        if (dir == null || !dir.exists()) return null;
        File[] files = dir.listFiles();
        if (files == null) return null;
        for (File f : files) {
            if (f.isDirectory()) {
                File r = findFileByExactName(f, exact);
                if (r != null) return r;
            } else if (f.getName().equals(exact)) {
                return f;
            }
        }
        return null;
    }
}
