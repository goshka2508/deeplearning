/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diplom.request.signclassification;

import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import javax.imageio.ImageIO;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author lucifer
 */
public class TrainingDialog extends javax.swing.JDialog {

    private MultiLayerNetwork network;

    public MultiLayerNetwork getNetwork() {
        return network;
    }

    /**
     * Creates new form TrainingDialog
     */
    public TrainingDialog(java.awt.Frame parent, boolean modal) {
        super(parent, modal);
        initComponents();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jLabel1 = new javax.swing.JLabel();
        iterations = new javax.swing.JTextField();
        jLabel2 = new javax.swing.JLabel();
        epochs = new javax.swing.JTextField();
        progress = new javax.swing.JProgressBar();
        cancel = new javax.swing.JButton();
        train = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
        setModalityType(java.awt.Dialog.ModalityType.APPLICATION_MODAL);
        setResizable(false);

        jLabel1.setText("Колличество итераций");

        iterations.setText("1");
        iterations.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                iterationsActionPerformed(evt);
            }
        });

        jLabel2.setText("Количество эпох");

        epochs.setText("1");

        cancel.setText("Отмена");
        cancel.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cancelActionPerformed(evt);
            }
        });

        train.setText("Обучить сеть");
        train.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                trainActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(iterations)
                    .addComponent(epochs)
                    .addComponent(progress, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(train, javax.swing.GroupLayout.PREFERRED_SIZE, 183, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(cancel, javax.swing.GroupLayout.DEFAULT_SIZE, 183, Short.MAX_VALUE))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jLabel1)
                            .addComponent(jLabel2))
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jLabel1)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(iterations, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jLabel2)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(epochs, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 24, Short.MAX_VALUE)
                .addComponent(progress, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cancel)
                    .addComponent(train))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void iterationsActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_iterationsActionPerformed

    }//GEN-LAST:event_iterationsActionPerformed

    private void cancelActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cancelActionPerformed
        dispose();
    }//GEN-LAST:event_cancelActionPerformed


    private void trainActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_trainActionPerformed

        ExecutorService es = ((TeacheableMachine) getParent()).getExec();

        final int epc = Integer.parseInt(epochs.getText());

        es.submit(new Runnable() {
            @Override
            public void run() {
                train.setEnabled(false);
                cancel.setEnabled(false);
                File dir = new File("images");
                if (dir.exists()) {
                    removeAllFiles(dir);
                }

                dir.mkdir();

                List<URI> images = new ArrayList<>();

                List<ClassificatorPanel> cp = ((TeacheableMachine) getParent()).getCls();
                for (ClassificatorPanel c : cp) {

                    File sub = new File(dir, c.getId());
                    sub.mkdir();
                    int id = 1;
                    for (BufferedImage bi : c.getSamples()) {
                        try {
                            File im = new File(sub, id + ".png");
                            ImageIO.write(bi, "png", im);
                            images.add(im.toURI());
                            id++;
                        } catch (Exception e) {
                        }
                    }
                }
                try {

                    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
                    CollectionInputSplit files = new CollectionInputSplit(images);

                    DataNormalization scaler = new ImagePreProcessingScaler(0, 1, 8);
                    ImageRecordReader recordReader = new ImageRecordReader(NNUtils.NN_HEIGHT, NNUtils.NN_WIDTH, NNUtils.NN_CHANNELS, labelMaker);
                    DataSetIterator dataIter;
                    recordReader.initialize(files);

                    dataIter = new RecordReaderDataSetIterator(recordReader, images.size(), 1, cp.size());
                    scaler.fit(dataIter);

                    dataIter.setPreProcessor(scaler);
                    MultiLayerNetwork model;
                    if (network == null) {
                        model = NNUtils.createNetworkLeNet(Integer.parseInt(iterations.getText()), ((TeacheableMachine) getParent()).getCls().size());
                        model.init();
                    } else {
                        model = network;
                    }

                    UIServer uiServer = UIServer.getInstance();

                    for (StatsStorage ss : new ArrayList<>(uiServer.getStatsStorageInstances())) {
                        try {
                            if (!ss.isClosed()) {
                                ss.close();
                            }
                            uiServer.disableRemoteListener();
                            uiServer.detach(ss);
                            uiServer.enableRemoteListener();
                        } catch (Exception e) {
                        }
                    }

                    StatsStorage statsStorage = new InMemoryStatsStorage();
                    uiServer.attach(statsStorage);
                    final long maxIters = Integer.parseInt(iterations.getText()) * epc;
                    model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1), new IterationListener() {

                        private boolean invoked = false;
                        private long iterCount = 0;

                        @Override
                        public boolean invoked() {
                            return invoked;
                        }

                        @Override
                        public void invoke() {
                            this.invoked = true;
                        }

                        @Override
                        public void iterationDone(Model model, int iteration) {
                            invoke();
                            progress.setValue((int) (iterCount * 100 / (maxIters)));
                            iterCount++;
                        }
                    });

                    for (int i = 0; i < epc; i++) {
                        model.fit(dataIter);
                        System.out.println("Epcoh " + (i + 1) + " of " + epc + " ...");

                    }

                    network = model;
                    ((TeacheableMachine) getParent()).setNetwork(network);

//                    uiServer.stop();
                } catch (Exception e) {
                    e.printStackTrace();
                }
                train.setEnabled(true);
                cancel.setEnabled(true);
                progress.setValue(0);
                train.setText("Продолжить обучение");

            }

            private void removeAllFiles(File dir) {
                for (File file : dir.listFiles()) {
                    if (file.isDirectory()) {
                        removeAllFiles(file);
                    } else {
                        file.delete();
                    }
                }
                dir.delete();
            }
        });

    }//GEN-LAST:event_trainActionPerformed

    /**
     * @param args the command line arguments
     */
//    public static void main(String args[]) {
//        /* Set the Nimbus look and feel */
//        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
//        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
//         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
//         */
//        try {
//            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
//                if ("Nimbus".equals(info.getName())) {
//                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
//                    break;
//                }
//            }
//        } catch (ClassNotFoundException ex) {
//            java.util.logging.Logger.getLogger(TrainingDialog.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
//        } catch (InstantiationException ex) {
//            java.util.logging.Logger.getLogger(TrainingDialog.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
//        } catch (IllegalAccessException ex) {
//            java.util.logging.Logger.getLogger(TrainingDialog.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
//        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
//            java.util.logging.Logger.getLogger(TrainingDialog.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
//        }
//        //</editor-fold>
//
//        /* Create and display the dialog */
//        java.awt.EventQueue.invokeLater(new Runnable() {
//            public void run() {
//                TrainingDialog dialog = new TrainingDialog(new javax.swing.JFrame(), true);
//                dialog.addWindowListener(new java.awt.event.WindowAdapter() {
//                    @Override
//                    public void windowClosing(java.awt.event.WindowEvent e) {
//                        System.exit(0);
//                    }
//                });
//                dialog.setVisible(true);
//            }
//        });
//    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton cancel;
    private javax.swing.JTextField epochs;
    private javax.swing.JTextField iterations;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JProgressBar progress;
    private javax.swing.JButton train;
    // End of variables declaration//GEN-END:variables
}
