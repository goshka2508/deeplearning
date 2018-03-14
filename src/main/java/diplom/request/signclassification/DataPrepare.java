/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diplom.request.signclassification;

import com.github.jaiimageio.impl.plugins.pnm.PNMImageReader;
import com.github.jaiimageio.impl.plugins.pnm.PNMImageReaderSpi;
import com.github.jaiimageio.impl.plugins.pnm.PNMImageWriter;
import com.github.jaiimageio.impl.plugins.pnm.PNMImageWriterSpi;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import javax.imageio.ImageIO;

/**
 *
 * @author lucifer
 */
public class DataPrepare {

    private static final int SIZE = 64;

    private static List<File> readCsv(String path, String name) {
        List<File> ret = new ArrayList<>();

        String line;
        int tg = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(path + "/GT-" + name + ".csv"))) {
            br.readLine();

            while ((line = br.readLine()) != null) {

                String[] s = line.split(";");

                String fname = s[0];
                File f = new File(path + "/rest" + fname);

//                if (f.exists()) {
//                    continue;
//                }
                int w = Integer.parseInt(s[1]);
                int h = Integer.parseInt(s[2]);
                int l = Integer.parseInt(s[3]);
                int t = Integer.parseInt(s[4]);
                int r = Integer.parseInt(s[5]);
                int b = Integer.parseInt(s[6]);

                w = r - l;
                h = b - t;

//                System.out.println(w + "x" + h);
//                if (w < 32 || h < 32) {
//                    continue;
//                }

                if (w != h) {
                    continue;
                }
                ret.add(f);
//                PNMImageReader pi = new PNMImageReader(new PNMImageReaderSpi());
//                pi.setInput(ImageIO.createImageInputStream(new FileInputStream(path + "/" + fname)));
//                BufferedImage im = pi.read(0);
//                pi.dispose();
//
//                im = im.getSubimage(l, t, r - l, b - t);
//
//                PNMImageWriter p = new PNMImageWriter(new PNMImageWriterSpi());
//                p.setOutput(ImageIO.createImageOutputStream(new FileOutputStream(f)));
//                p.write(im);
//                p.dispose();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        Collections.sort(ret);

        while (ret.size() > SIZE) {
            int step = ret.size() - SIZE;
            int s = step / SIZE;//ret.size();
            s = s == 0 ? 1 : s;
            step = 0;
            for (int i = ret.size() - 1; i > 0; i -= 1) {
                if (step < s) {
                    step++;
                    ret.remove(i);
                } else {
                    step = 0;
                }
                if (ret.size() <= SIZE) {
                    break;
                }
            }
        }

        System.out.println(name + " -> " + ret.size());

        return ret;
    }

    private static Map<String, List<File>> getFiles(String path) {
        Map<String, List<File>> ret = new HashMap<>();

        File p = new File(path);
        if (!p.isDirectory()) {
            return ret;
        }

        int fv = 0;
        for (File f : p.listFiles()) {
            ret.put(UUID.randomUUID().toString(), readCsv(path + "/" + f.getName(), f.getName()));

        }
        return ret;
    }

    public static void main(String[] args) throws Exception {
        Map<String, List<File>> files = getFiles("H:\\JAVA\\GTSRB\\Final_Training\\Images");

        ZipOutputStream zos = new ZipOutputStream(new FileOutputStream("signs.nnb"));

        for (String key : files.keySet()) {
            if (files.get(key).size() < 16) {
                continue;
            }
            int id = 0;
            for (File f : files.get(key)) {
                ZipEntry ze = new ZipEntry(key + "/" + id + ".png");
                id++;

                zos.putNextEntry(ze);
                PNMImageReader pi = new PNMImageReader(new PNMImageReaderSpi());
                pi.setInput(ImageIO.createImageInputStream(new FileInputStream(f)));
                BufferedImage im = pi.read(0);
                pi.dispose();

                ImageIO.write(im, "PNG", zos);

                zos.closeEntry();
            }

        }

        zos.close();
    }
}
