/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diplom.request.signclassification;

import com.github.sarxos.webcam.Webcam;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.RescaleOp;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import javax.imageio.ImageIO;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.filechooser.FileNameExtensionFilter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
//import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author lucifer
 */
public class TeacheableMachine extends javax.swing.JFrame {

    private final ExecutorService exec = Executors.newCachedThreadPool();
    private boolean isStarted = true;
    private FlowPanel fp;
    private List<ClassificatorPanel> cls = new ArrayList<>();

    /**
     * Creates new form TeacheableMachine
     */
    public TeacheableMachine() {
        initComponents();
        initCameraThread();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        camera = new ImagePanel();
        jPanel3 = new javax.swing.JPanel();
        addClass = new javax.swing.JButton();
        clearNetwork = new javax.swing.JButton();
        trainNetwork = new javax.swing.JButton();
        testNetwork = new javax.swing.JButton();
        saveNetwork = new javax.swing.JButton();
        readNetwork = new javax.swing.JButton();
        classPanel = new javax.swing.JPanel();
        classificators = new javax.swing.JScrollPane(fp = new FlowPanel());

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jPanel1.setBorder(javax.swing.BorderFactory.createTitledBorder("Камера"));

        camera.setPreferredSize(new java.awt.Dimension(256, 256));

        javax.swing.GroupLayout cameraLayout = new javax.swing.GroupLayout(camera);
        camera.setLayout(cameraLayout);
        cameraLayout.setHorizontalGroup(
            cameraLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 256, Short.MAX_VALUE)
        );
        cameraLayout.setVerticalGroup(
            cameraLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 256, Short.MAX_VALUE)
        );

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(camera, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addComponent(camera, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 8, Short.MAX_VALUE))
        );

        jPanel3.setBorder(javax.swing.BorderFactory.createTitledBorder("Действия"));

        addClass.setText("Добавить класс");
        addClass.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                addClassActionPerformed(evt);
            }
        });

        clearNetwork.setText("Сбросить параметры нейронной сети");
        clearNetwork.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                clearNetworkActionPerformed(evt);
            }
        });

        trainNetwork.setText("Обучить сеть");
        trainNetwork.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                trainNetworkActionPerformed(evt);
            }
        });

        testNetwork.setText("Тестировать сеть");
        testNetwork.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                testNetworkActionPerformed(evt);
            }
        });

        saveNetwork.setText("Записать сеть");
        saveNetwork.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                saveNetworkActionPerformed(evt);
            }
        });

        readNetwork.setText("Прочитать сеть");
        readNetwork.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                readNetworkActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout jPanel3Layout = new javax.swing.GroupLayout(jPanel3);
        jPanel3.setLayout(jPanel3Layout);
        jPanel3Layout.setHorizontalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel3Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(addClass, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(clearNetwork, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(trainNetwork, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(testNetwork, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(saveNetwork, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(readNetwork, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        jPanel3Layout.setVerticalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel3Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(addClass)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(clearNetwork)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(trainNetwork)
                .addGap(18, 18, 18)
                .addComponent(testNetwork)
                .addGap(18, 18, 18)
                .addComponent(saveNetwork)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(readNetwork)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        classPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Классификатор"));

        javax.swing.GroupLayout classPanelLayout = new javax.swing.GroupLayout(classPanel);
        classPanel.setLayout(classPanelLayout);
        classPanelLayout.setHorizontalGroup(
            classPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(classPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(classificators, javax.swing.GroupLayout.DEFAULT_SIZE, 598, Short.MAX_VALUE)
                .addContainerGap())
        );
        classPanelLayout.setVerticalGroup(
            classPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(classPanelLayout.createSequentialGroup()
                .addComponent(classificators)
                .addContainerGap())
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(jPanel3, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(classPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(classPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jPanel3, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(0, 55, Short.MAX_VALUE)))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void trainNetworkActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_trainNetworkActionPerformed
        TrainingDialog tr = new TrainingDialog(this, true);
        tr.setVisible(true);
    }//GEN-LAST:event_trainNetworkActionPerformed

    private boolean testingEnabled = false;

    private void testNetworkActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_testNetworkActionPerformed

        if (network == null) {
            return;
        }

        if (testingEnabled) {

            addClass.setEnabled(true);
            clearNetwork.setEnabled(true);
            trainNetwork.setEnabled(true);
            saveNetwork.setEnabled(true);
            readNetwork.setEnabled(true);

            testingEnabled = false;
            return;
        }
        addClass.setEnabled(false);
        clearNetwork.setEnabled(false);
        trainNetwork.setEnabled(false);
        saveNetwork.setEnabled(false);
        readNetwork.setEnabled(false);

        testingEnabled = true;

        exec.submit(new Runnable() {
            @Override
            public void run() {

                NativeImageLoader loader = new NativeImageLoader(NNUtils.NN_HEIGHT, NNUtils.NN_WIDTH, NNUtils.NN_CHANNELS);
                DataNormalization scaler = new ImagePreProcessingScaler(0, 1, 8);
                while (testingEnabled) {
                    try {
                        BufferedImage img = ((ImagePanel) camera).getImg();
                        BufferedImage gg = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_BYTE_GRAY);

                        RescaleOp rescaleOp = new RescaleOp(1.5f, 15, null);
                        rescaleOp.filter(gg, gg);

                        gg.getGraphics().drawImage(img, 0, 0, null);
                        INDArray image = loader.asMatrix(gg);
                        scaler.transform(image);
                        INDArray predictions = network.output(image);
//                        for (ClassificatorPanel cp : cls) {
//                            cp.getScore().setValue(0);
//                        }

                        List<String> labels = new ArrayList<>();
                        for (ClassificatorPanel cp : cls) {
                            labels.add(cp.getId());
                        }

                        Collections.sort(labels);

                        for (int batch = 0; batch < predictions.size(0); batch++) {
                            INDArray currentBatch = predictions.getRow(batch).dup();
                            int i = 0;
                            while (i < cls.size()) {
                                int num = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
                                float score = currentBatch.getFloat(batch, num);
                                currentBatch.putScalar(0, num, 0);

                                for (ClassificatorPanel cp : cls) {
                                    if (cp.getId().equals(labels.get(num))) {
                                        cp.getScore().setValue((int) (score * 100));
                                    }
                                }
                                i++;
                            }
                        }
                    } catch (Exception e) {
                    }
                }

            }
        });

    }//GEN-LAST:event_testNetworkActionPerformed

    private void addClassActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_addClassActionPerformed
        ClassificatorPanel cp = new ClassificatorPanel(this);
        cp.setID(UUID.randomUUID().toString());

        addClass(cp);


    }//GEN-LAST:event_addClassActionPerformed

    private void saveNetworkActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_saveNetworkActionPerformed

        JFileChooser fc = new JFileChooser();
        FileNameExtensionFilter filter = new FileNameExtensionFilter("Данные нейронной сети", "nnb");
        fc.setFileFilter(filter);
        int returnVal = fc.showSaveDialog(this);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            File f = fc.getSelectedFile();
            if (!f.getName().endsWith(".nnb")) {
                f = new File(f.getAbsolutePath() + ".nnb");
            }
            try {
                ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(f), Charset.forName("UTF-8"));

                if (network != null) {
                    ZipEntry ze = new ZipEntry("network.model");
                    zos.putNextEntry(ze);
                    ModelSerializer.writeModel(network, zos, false);
                    zos.closeEntry();
                }

                for (ClassificatorPanel cp : cls) {
                    int idx = 1;
                    String id = cp.getId();
                    for (BufferedImage bi : cp.getSamples()) {
                        ZipEntry ze = new ZipEntry(id + "/" + idx + ".png");

                        zos.putNextEntry(ze);
                        ImageIO.write(bi, "PNG", zos);
                        zos.closeEntry();

                        idx++;
                    }
                }

                zos.close();
            } catch (Throwable e) {
                e.printStackTrace();
            }
        }
    }//GEN-LAST:event_saveNetworkActionPerformed

    private void readNetworkActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_readNetworkActionPerformed

        JFileChooser fc = new JFileChooser();
        FileNameExtensionFilter filter = new FileNameExtensionFilter("Данные нейронной сети", "nnb");
        fc.setFileFilter(filter);

        int returnVal = fc.showOpenDialog(this);
        if (returnVal == JFileChooser.APPROVE_OPTION) {

            cls.forEach((cp) -> {
                fp.remove(cp);
            });
            cls.clear();
            if (network != null) {
                network.clear();
                network.clearLayerMaskArrays();
            }
            network = null;

            File f = fc.getSelectedFile();
            try {
                ZipInputStream zis = new ZipInputStream(new FileInputStream(f));

                while (true) {

                    ZipEntry ze = zis.getNextEntry();
                    if (ze == null) {
                        break;
                    }

                    if (ze.getName().equals("network.model")) {
                        network = ModelSerializer.restoreMultiLayerNetwork(zis);
                    }

                    if (ze.getName().endsWith(".png")) {
                        String id = ze.getName().substring(0, ze.getName().indexOf("/"));
                        ClassificatorPanel cp = null;
                        boolean contains = false;

                        for (ClassificatorPanel c : cls) {
                            if (c.getId().equals(id)) {
                                cp = c;
                                contains = true;
                                break;
                            }
                        }
                        if (!contains) {
                            cp = new ClassificatorPanel(this);
                            cp.setID(id);
                            addClass(cp);
                        }

                        cp.addSample(ImageIO.read(zis));
                    }

                    zis.closeEntry();
                }
                zis.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }//GEN-LAST:event_readNetworkActionPerformed

    private void clearNetworkActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_clearNetworkActionPerformed
        if (network != null) {
            network.clear();
            network.clearLayerMaskArrays();
        }
        network = null;

        cls.stream().map((cp) -> {
            cp.getSamples().clear();
            return cp;
        }).forEachOrdered((cp) -> {
            fp.remove(cp);
        });

        cls.clear();
        
        fp.updateUI();
    }//GEN-LAST:event_clearNetworkActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
//        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);
//        CudaEnvironment.getInstance().getConfiguration()
//                .allowMultiGPU(true)
//                .setMaximumDeviceCacheableLength(1024 * 1024 * 1024L)
//                .setMaximumDeviceCache(4L * 1024 * 1024 * 1024L)
//                .setMaximumHostCacheableLength(1024 * 1024 * 1024L)
//                .setMaximumHostCache(4L * 1024 * 1024 * 1024L)
//                .allowCrossDeviceAccess(true);

        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(TeacheableMachine.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(TeacheableMachine.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(TeacheableMachine.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(TeacheableMachine.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new TeacheableMachine().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton addClass;
    private javax.swing.JPanel camera;
    private javax.swing.JPanel classPanel;
    private javax.swing.JScrollPane classificators;
    private javax.swing.JButton clearNetwork;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel3;
    private javax.swing.JButton readNetwork;
    private javax.swing.JButton saveNetwork;
    private javax.swing.JButton testNetwork;
    private javax.swing.JButton trainNetwork;
    // End of variables declaration//GEN-END:variables

    private void initCameraThread() {
        final Webcam webcam = Webcam.getDefault();
        webcam.setViewSize(new Dimension(640, 480));
        webcam.open();

        exec.submit(new Runnable() {
            @Override
            public void run() {
                while (isStarted) {
                    try {
                        BufferedImage img = new BufferedImage(256, 256, BufferedImage.TYPE_INT_ARGB);
                        BufferedImage im = webcam.getImage();
                        im = im.getSubimage((im.getWidth() - im.getHeight()) / 2, 0, im.getHeight(), im.getHeight());
                        ((Graphics2D) img.getGraphics()).drawImage(im, 0, 0, img.getHeight(), img.getWidth(), 0, 0, im.getHeight(), im.getWidth(), null);

                        ((ImagePanel) camera).setImg(img);

                        SwingUtilities.invokeAndWait(new Runnable() {
                            @Override
                            public void run() {
                                camera.repaint();
                            }
                        });
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                }
            }
        });
    }

    protected void removeClass(ClassificatorPanel cp) {
        fp.remove(cp);
        cls.remove(cp);
        updateClassificators();
    }

    protected void addClass(ClassificatorPanel cp) {
        fp.add(cp);
        cls.add(cp);
        updateClassificators();
    }

    private void updateClassificators() {

        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                classificators.updateUI();
                classificators.repaint();
            }
        });
    }

    ImagePanel getCamera() {
        return (ImagePanel) camera;
    }

    public ExecutorService getExec() {
        return exec;
    }

    List<ClassificatorPanel> getCls() {
        return cls;
    }

    MultiLayerNetwork network;

    void setNetwork(MultiLayerNetwork network) {
        this.network = network;
    }

}
