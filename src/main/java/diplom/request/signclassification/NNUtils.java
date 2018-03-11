/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diplom.request.signclassification;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author lucifer
 */
public class NNUtils {

    public final static int NN_WIDTH = 64;
    public final static int NN_HEIGHT = 64;
    public final static int NN_CHANNELS = 3;

    public static MultiLayerNetwork createNetworkLeNet(int iterations, int outputs) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.SINGLE)
                .inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                .seed(12312345)
                .iterations(iterations)
                .activation(Activation.IDENTITY)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new AdaDelta())
                .convolutionMode(ConvolutionMode.Same)
                .list()
                .layer(0, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1})
                        .name("cnn1")
                        .nIn(3)
                        .nOut(10)
                        .activation(Activation.RELU)
                        .build()
                )
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2},
                        new int[]{2, 2}).name("maxpool1").build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}).name("cnn2").nOut(25)
                        .activation(Activation.RELU).build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2},
                        new int[]{2, 2}).name("maxpool2").build())
                .layer(4, new DenseLayer.Builder().name("ffn1").activation(Activation.RELU).nOut(100).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).name("output")
                        .nOut(outputs).activation(Activation.SOFTMAX) // radial basis function required
                        .build())
                .setInputType(InputType.convolutionalFlat(NN_HEIGHT, NN_WIDTH, NN_CHANNELS))
                .backprop(true).pretrain(false).build();

        return new MultiLayerNetwork(conf);
    }

    public static MultiLayerNetwork createNetworkAlexNet(int iterations, int outputs) {
        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(1231234)
                .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(iterations)
                .updater(new Nesterovs(1e-2, 0.9))
                //                .biasUpdater(new Nesterovs(2e-2, 0.9))
                .convolutionMode(ConvolutionMode.Same)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                .dropOut(0.5).l2(5 * 1e-4).miniBatch(false)
                .list().layer(0,
                        new ConvolutionLayer.Builder(new int[]{11, 11}, new int[]{4, 4},
                        new int[]{2, 2}).name("cnn1")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Truncate)
                        .nIn(3).nOut(64).build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                        new int[]{2, 2}, new int[]{1, 1}).convolutionMode(ConvolutionMode.Truncate)
                        .name("maxpool1").build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{2, 2}, new int[]{1, 1}) // TODO: fix input and put stride back to 1,1
                        .convolutionMode(ConvolutionMode.Truncate).name("cnn2")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).nOut(192)
                        .biasInit(nonZeroBias).build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                        new int[]{2, 2}).name("maxpool2").build())
                .layer(4, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn3").cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).nOut(384)
                        .build())
                .layer(5, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn4").cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).nOut(256)
                        .biasInit(nonZeroBias).build())
                .layer(6, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1})
                        .name("cnn5").cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).nOut(256)
                        .biasInit(nonZeroBias).build())
                .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3},
                        new int[]{2, 2}) // TODO: fix input and put stride back to 2,2
                        .name("maxpool3").build())
                .layer(8, new DenseLayer.Builder().name("ffn1").nIn(256).nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005)).biasInit(nonZeroBias).dropOut(dropOut)
                        .build())
                .layer(9, new DenseLayer.Builder().name("ffn2").nOut(4096)
                        .dist(new GaussianDistribution(0, 0.005)).biasInit(nonZeroBias).dropOut(dropOut)
                        .build())
                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output").nOut(outputs).activation(Activation.SOFTMAX).build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutionalFlat(NN_HEIGHT, NN_WIDTH, NN_CHANNELS)).build();

        return new MultiLayerNetwork(conf);
    }

    public static MultiLayerNetwork createNetworkSimple(int iterations, int outputs) {
        int seed = 123123;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.005)
                .learningRate(0.001)
                .weightInit(WeightInit.XAVIER)
                //.activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(3)
                        .stride(1, 1)
                        .nOut(8)
                        .biasInit(0)
                        .activation(Activation.IDENTITY)
                        .build()
                )
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build()
                )
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(16)
                        .biasInit(0)
                        .activation(Activation.IDENTITY)
                        .build()
                )
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build()
                )
                .layer(4, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(64)
                        .biasInit(0)
                        .activation(Activation.IDENTITY)
                        .build()
                ).layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build()
                )
                .layer(6, new DenseLayer.Builder()
                        .activation(Activation.IDENTITY)
                        .nOut(500)
                        .build()
                )
                .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputs)
                        .activation(Activation.SOFTMAX)
                        .build()
                )
                .setInputType(InputType.convolutionalFlat(NN_HEIGHT, NN_WIDTH, NN_CHANNELS))
                .backprop(true).pretrain(false).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        return model;
    }
}
