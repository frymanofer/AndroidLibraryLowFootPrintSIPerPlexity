package ai.perplexity.hotword.speakerid;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;

import java.io.*;

public final class SpeakerIdStorage {
    private SpeakerIdStorage() {}

    /** Default app-internal file paths. */
    public static File defaultMeanEmbFile(Context ctx) {
        return new File(ctx.getFilesDir(), "speaker_emb.npy");
    }
    public static File defaultMeanCountFile(Context ctx) {
        return new File(ctx.getFilesDir(), "speaker_emb.npy.count");
    }
    public static File defaultClusterFile(Context ctx) {
        return new File(ctx.getFilesDir(), "speaker_emb_cluster.npy");
    }

    /** Returns true if both mean and cluster files exist. */
    public static boolean hasDefaultTargets(Context ctx) {
        return defaultMeanEmbFile(ctx).exists() && defaultClusterFile(ctx).exists();
    }

    /** Copy a file to the public Downloads folder, returning its URI. */
    public static Uri exportToDownloads(Context ctx, File src, String displayName) throws IOException {
        if (!src.exists()) throw new FileNotFoundException(src.getAbsolutePath());

        if (Build.VERSION.SDK_INT >= 29) {
            ContentResolver cr = ctx.getContentResolver();
            ContentValues cv = new ContentValues();
            cv.put(MediaStore.Downloads.DISPLAY_NAME, displayName);
            cv.put(MediaStore.Downloads.MIME_TYPE, "application/octet-stream");
            cv.put(MediaStore.Downloads.IS_PENDING, 1);
            Uri col = MediaStore.Downloads.EXTERNAL_CONTENT_URI;
            Uri uri = cr.insert(col, cv);
            if (uri == null) throw new IOException("Failed to insert into MediaStore");

            try (OutputStream os = cr.openOutputStream(uri);
                 InputStream is = new FileInputStream(src)) {
                byte[] buf = new byte[8192];
                int n;
                while ((n = is.read(buf)) > 0) os.write(buf, 0, n);
            }

            cv.clear();
            cv.put(MediaStore.Downloads.IS_PENDING, 0);
            cr.update(uri, cv, null, null);
            return uri;
        } else {
            // Legacy external storage
            File dst = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), displayName);
            try (InputStream is = new FileInputStream(src);
                 OutputStream os = new FileOutputStream(dst)) {
                byte[] buf = new byte[8192];
                int n;
                while ((n = is.read(buf)) > 0) os.write(buf, 0, n);
            }
            return Uri.fromFile(dst);
        }
    }
    public static void wipeDefaults(Context ctx) {
        try { defaultMeanEmbFile(ctx).delete(); } catch (Exception ignore) {}
        try { defaultMeanCountFile(ctx).delete(); } catch (Exception ignore) {}
        try { defaultClusterFile(ctx).delete(); } catch (Exception ignore) {}
    }
}
