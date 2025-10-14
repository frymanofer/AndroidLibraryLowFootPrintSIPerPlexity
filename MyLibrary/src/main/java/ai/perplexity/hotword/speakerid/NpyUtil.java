package ai.perplexity.hotword.speakerid;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import android.util.Log;
import java.nio.channels.FileChannel;


/** Minimal NumPy .npy v1.0 writer/reader for float32 (1D and 2D). */
public final class NpyUtil {
    
    private static final String TAG = "NpyUtil";

    // === NEW: atomic wrappers ===
    public static void saveVectorFloat32Atomic(File f, float[] v) throws IOException {
        saveArrayFloat32Atomic(f, new float[][]{v});
    }

        private static void saveArrayFloat32Atomic(File dest, float[][] m) throws IOException {
        File dir = dest.getParentFile();
        if (dir != null && !dir.exists()) dir.mkdirs();
        File tmp = File.createTempFile(dest.getName(), ".tmp", dir);

        // Write to temp
        writeArrayFloat32(tmp, m);

        // Force to disk
        try (FileOutputStream fos = new FileOutputStream(tmp, true)) {
            FileChannel ch = fos.getChannel();
            ch.force(true); // metadata + data
            fos.getFD().sync();
        } catch (Throwable t) {
            //noinspection ResultOfMethodCallIgnored
            tmp.delete();
            throw new IOException("fsync failed for " + tmp, t);
        }

        // Atomic replace
        if (!tmp.renameTo(dest)) {
            // fallback: delete dest then rename
            //noinspection ResultOfMethodCallIgnored
            dest.delete();
            if (!tmp.renameTo(dest)) {
                //noinspection ResultOfMethodCallIgnored
                tmp.delete();
                throw new IOException("renameTo failed for " + dest);
            }
        }

        // Validate by re-reading
        if (m.length == 1) {
            float[] v = loadVectorFloat32(dest);
            validateVector(v, "vector " + dest.getName());
        } else {
            float[][] mm = loadMatrixFloat32(dest);
            validateMatrix(mm, "matrix " + dest.getName(), m[0].length);
        }
        Log.i(TAG, "saveArrayFloat32Atomic: wrote+validated " + dest.getAbsolutePath());
    }

    // === existing save, split so atomic wrapper can reuse ===
    private static void writeArrayFloat32(File f, float[][] m) throws IOException {
        int rows = m.length;
        int cols = m[0].length;
        for (int i = 1; i < rows; ++i)
            if (m[i].length != cols) throw new IllegalArgumentException("Jagged array not supported");

        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(f)))) {
            out.writeBytes("\u0093NUMPY");
            out.writeByte(0x01);
            out.writeByte(0x00);

            String header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                    rows + (rows == 1 ? "," : "," + cols) + "), }";
            int base = 12; // 10 magic+ver + 2 length
            int headerLen = header.getBytes(StandardCharsets.US_ASCII).length;
            int padLen = (16 - ((base + headerLen) % 16)) % 16;
            int totalHeader = headerLen + padLen;

            out.writeShort(Short.reverseBytes((short) totalHeader));
            out.write(header.getBytes(StandardCharsets.US_ASCII));
            for (int i = 0; i < padLen; ++i) out.writeByte(' ');

            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                    out.writeInt(Integer.reverseBytes(Float.floatToRawIntBits(m[r][c])));
            out.flush();
        }
    }

    // === NEW: validators ===
    private static void validateVector(float[] v, String tag) throws IOException {
        if (v == null || v.length == 0) throw new IOException(tag + " empty");
        for (float x : v) {
            if (!Float.isFinite(x)) throw new IOException(tag + " has non-finite");
        }
    }

    private static void validateMatrix(float[][] m, String tag, int expectCols) throws IOException {
        if (m == null || m.length == 0) throw new IOException(tag + " empty rows");
        for (float[] row : m) {
            if (row == null || row.length == 0) throw new IOException(tag + " empty row");
            if (expectCols > 0 && row.length != expectCols)
                throw new IOException(tag + " wrong cols: " + row.length + " vs " + expectCols);
            for (float x : row) if (!Float.isFinite(x)) throw new IOException(tag + " has non-finite");
        }
    }

    public static void saveMatrixFloat32Atomic(File f, float[][] m) throws IOException {
        saveArrayFloat32Atomic(f, m);
    }

    private NpyUtil() {}

    public static void saveVectorFloat32(File f, float[] v) throws IOException {
        saveArrayFloat32(f, new float[][]{v});
    }

    public static void saveMatrixFloat32(File f, float[][] m) throws IOException {
        saveArrayFloat32(f, m);
    }

    private static void saveArrayFloat32(File f, float[][] m) throws IOException {
        // shape: (rows, cols) ; rows may be 1 for vector
        int rows = m.length;
        int cols = m[0].length;
        for (int i = 1; i < rows; ++i)
            if (m[i].length != cols) throw new IllegalArgumentException("Jagged array not supported");

        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(f)))) {
            out.writeBytes("\u0093NUMPY"); // magic (0x93,'NUMPY')
            out.writeByte(0x01); // major
            out.writeByte(0x00); // minor

            String header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                    rows + (rows == 1 ? "," : "") + (cols == 0 ? "" : "," + cols) + "), }";
            // npy v1.0: 10-byte preamble, then 2-byte header-len little endian, then padded header to 16-byte alignment
            int base = 10 + 2;
            int headerLen = header.getBytes(StandardCharsets.US_ASCII).length;
            int padLen = (16 - ((base + headerLen) % 16)) % 16;
            int totalHeader = headerLen + padLen;

            out.writeShort(Short.reverseBytes((short) totalHeader));
            out.write(header.getBytes(StandardCharsets.US_ASCII));
            for (int i = 0; i < padLen; ++i) out.writeByte(' ');

            // data
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                    out.writeInt(Integer.reverseBytes(Float.floatToRawIntBits(m[r][c])));
        }
    }

    /** Load float32 vector from .npy (supports 1D or 2D where rows==1). */
    public static float[] loadVectorFloat32(File f) throws IOException {
        float[][] mat = loadMatrixFloat32(f);
        if (mat.length == 1) return mat[0];
        if (mat[0].length == 1) {
            float[] v = new float[mat.length];
            for (int i = 0; i < mat.length; ++i) v[i] = mat[i][0];
            return v;
        }
        // If KxD file given, return the mean row (useful fallback)
        int rows = mat.length, cols = mat[0].length;
        float[] mean = new float[cols];
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) mean[c] += mat[r][c];
        for (int c = 0; c < cols; ++c) mean[c] /= rows;
        return mean;
    }

    /** Load float32 2D from .npy (v1); throws for unsupported headers. */
    public static float[][] loadMatrixFloat32(File f) throws IOException {
        if (f.length() < 64) throw new IOException("npy too small: " + f.length() + " bytes");
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(f)))) {
            byte[] magic = new byte[6];
            in.readFully(magic);
            if (!new String(magic, StandardCharsets.ISO_8859_1).equals("\u0093NUMPY"))
                throw new IOException("Not a .npy file");
            int major = in.readUnsignedByte();
            int minor = in.readUnsignedByte();
            if (!(major == 1 && minor == 0)) throw new IOException("Only .npy v1.0 supported");

            int hlen = Short.reverseBytes(in.readShort()) & 0xFFFF;
            byte[] hdr = new byte[hlen];
            in.readFully(hdr);
            String header = new String(hdr, StandardCharsets.US_ASCII).trim();

            if (!header.contains("'descr': '<f4'") || !header.contains("'fortran_order': False"))
                throw new IOException("Only little-endian float32, C-order supported");

            int[] shape = parseShape(header);
            int rows, cols;
            if (shape.length == 1) { rows = 1; cols = shape[0]; }
            else if (shape.length == 2) { rows = shape[0]; cols = shape[1]; }
            else throw new IOException("Unsupported shape: " + Arrays.toString(shape));

            float[][] out = new float[rows][cols];
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    int bits = Integer.reverseBytes(in.readInt());
                    out[r][c] = Float.intBitsToFloat(bits);
                }
            }
            return out;
        } catch (EOFException eof) {
            throw new IOException("npy truncated (EOF)", eof);
        }
    }

    private static int[] parseShape(String header) throws IOException {
        int i = header.indexOf("'shape':");
        if (i < 0) throw new IOException("shape not found");
        int lp = header.indexOf('(', i);
        int rp = header.indexOf(')', lp);
        String inside = header.substring(lp + 1, rp).trim();
        if (inside.isEmpty()) return new int[]{};
        String[] toks = inside.split(",");
        int n = 0;
        for (String t : toks) if (!t.trim().isEmpty()) n++;
        int[] shape = new int[n];
        int k = 0;
        for (String t : toks) {
            t = t.trim();
            if (t.isEmpty()) continue;
            shape[k++] = Integer.parseInt(t);
        }
        return shape;
    }
}
