package ai.perplexity.hotword.speakerid;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;
import java.io.*;
import java.util.*;

final class SpeakerIdAssets {
    private static final String TAG = "SpeakerIdAssets";

    static final class Paths {
        final String speakerOnnx;  // nemo_en_speakerverification_speakernet.onnx
        final String vadOnnx;      // silero_vad.onnx
        Paths(String speakerOnnx, String vadOnnx) {
            this.speakerOnnx = speakerOnnx; this.vadOnnx = vadOnnx;
        }
    }

    private static String findAssetPath(Context ctx, String fileName) {
        try {
            AssetManager am = ctx.getAssets();
            Deque<String> stack = new ArrayDeque<>();
            stack.push("");
            while (!stack.isEmpty()) {
                String dir = stack.pop();
                String[] entries = am.list(dir);
                if (entries == null) continue;
                for (String e : entries) {
                    String path = dir.isEmpty() ? e : (dir + "/" + e);
                    String[] sub = am.list(path);
                    boolean isDir = (sub != null && sub.length > 0);
                    if (isDir) stack.push(path);
                    else if (e.equalsIgnoreCase(fileName)) {
                        Log.d(TAG, "findAssetPath: found '" + fileName + "' at '" + path + "'");
                        return path;
                    }
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "findAssetPath error: " + e);
        }
        Log.d(TAG, "findAssetPath: NOT FOUND '" + fileName + "'");
        return null;
    }

    // ... inside class SpeakerIdAssets ...
    static String copyAssetIfExists(Context ctx, String assetName, String subdir) {
        try {
            String[] list = ctx.getAssets().list("");
            if (list != null) {
                for (String s : list) if (s.equals(assetName)) {
                    return copyAssetToFiles(ctx, assetName, subdir);
                }
            }
        } catch (Exception ignore) {}
        return null;
    }

    private static boolean assetExists(Context ctx, String fileName) {
        boolean ok = (findAssetPath(ctx, fileName) != null);
        Log.d(TAG, "assetExists(" + fileName + ") -> " + ok);
        return ok;
    }

    private static String copyAssetToFiles(Context ctx, String assetName, String subdir) {
        String assetPath = findAssetPath(ctx, assetName);
        if (assetPath == null) return null;
        File outDir = new File(ctx.getFilesDir(), subdir == null ? "speakerid" : subdir);
        outDir.mkdirs();
        File out = new File(outDir, assetName.substring(assetName.lastIndexOf('/') + 1));
        long total = 0;
        try (InputStream in = ctx.getAssets().open(assetPath);
             OutputStream os = new FileOutputStream(out)) {
            byte[] buf = new byte[16 * 1024]; int n;
            while ((n = in.read(buf)) >= 0) { os.write(buf, 0, n); total += n; }
            Log.i(TAG, "copyAssetToFiles: '" + assetName + "' -> " + out.getAbsolutePath() + " (" + total + " bytes)");
            return out.getAbsolutePath();
        } catch (Exception e) {
            Log.e(TAG, "copyAssetToFiles failed for " + assetName + ": " + e);
            return null;
        }
    }

    /** Strict mapping with heuristics fallback, using the SAME DM format as KeyWordsDetection. */
    static Paths resolveModels(Context ctx) {
        Log.i(TAG, "resolveModels: BEGIN");
        String speakernet = null;
        String vad = null;

        // 1) speaker_id.dm → nemo_en_speakerverification_speakernet.onnx
        if (assetExists(ctx, "speaker_id.dm")) {
            String dmPath = copyAssetToFiles(ctx, "speaker_id.dm", "speakerid");
            if (dmPath != null) {
                File outDir = new File(ctx.getFilesDir(), "speakerid_dm/speaker_id");
                Map<String,String> pick = DmExtractor.expandDmAndPick(dmPath, outDir,
                        "nemo_en_speakerverification_speakernet.onnx");
                speakernet = pick.get("nemo_en_speakerverification_speakernet.onnx");
                if (speakernet == null) {
                    // Heuristic fallback: first .onnx, or anything with "speakernet"/"speaker"
                    File f = DmExtractor.findFirstByHeuristic(outDir,
                            "speakernet", "speaker");
                    if (f != null) {
                        speakernet = f.getAbsolutePath();
                        Log.w(TAG, "resolveModels: speaker_id.dm heuristic chose " + f.getName());
                    } else {
                        Log.w(TAG, "resolveModels: speaker_id.dm has no .onnx");
                    }
                }
            }
        } else {
            Log.w(TAG, "resolveModels: speaker_id.dm not in assets");
        }

        // 2) layer1.dm → silero_vad.onnx
        if (assetExists(ctx, "layer1.dm")) {
            String dmPath = copyAssetToFiles(ctx, "layer1.dm", "speakerid");
            if (dmPath != null) {
                File outDir = new File(ctx.getFilesDir(), "layer1_dm/layer1");
                Map<String,String> pick = DmExtractor.expandDmAndPick(dmPath, outDir, "silero_vad.onnx");
                vad = pick.get("silero_vad.onnx");
                if (vad == null) {
                    File f = DmExtractor.findFirstByHeuristic(outDir, "vad", "silero");
                    if (f != null) {
                        vad = f.getAbsolutePath();
                        Log.w(TAG, "resolveModels: layer1.dm heuristic chose " + f.getName());
                    } else {
                        Log.w(TAG, "resolveModels: layer1.dm has no .onnx");
                    }
                }
            }
        } else {
            Log.w(TAG, "resolveModels: layer1.dm not in assets");
        }

        // Optional fallbacks to direct-assets
        if (speakernet == null && assetExists(ctx, "nemo_en_speakerverification_speakernet.onnx")) {
            speakernet = copyAssetToFiles(ctx, "nemo_en_speakerverification_speakernet.onnx", "speakerid");
            Log.w(TAG, "resolveModels: fallback to direct speakernet asset");
        }
        if (vad == null && assetExists(ctx, "silero_vad.onnx")) {
            vad = copyAssetToFiles(ctx, "silero_vad.onnx", "speakerid");
            Log.w(TAG, "resolveModels: fallback to direct vad asset");
        }

        Log.i(TAG, "resolveModels: DONE speakerOnnx=" + speakernet + " , vadOnnx=" + vad);
        return new Paths(speakernet, vad);
    }
}
