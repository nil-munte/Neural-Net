import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Locale;
import java.util.Scanner;

public class NeuralNet {
  public int L; // number of layers
  public int[] n; // an array with the number of units in each layer
  public double[][] h; // an array of arrays for the fields (h)
  public double[][] xi; // an array of arrays for the activations (ξ)
  public double[][][] w; // an array of matrices for the weights (w)
  public double[][] theta; // an array of arrays for the thresholds (θ)
  public double[][] delta; // an array of arrays for the propagation errors (Δ)
  public double[][][] d_w; // an array of matrices for the changes of the weights (δw)
  public double[][] d_theta; // an array of arrays for the changes of the thresholds (δθ)
  public double[][][] d_w_prev;   // an array of matrices for the previous changes of the weights, used for the
                                  // momentum therm (δw(prev))
  public double[][] d_theta_prev;   // an array of arrays for the previous changes of the thresholds, used for the
                                    // momentum term (δθ(prev))

  public ActivationFunction fact; // the name of the activation function (sigmoid,relu,linear, tanh)

  public NeuralNet(int[] layers) {
    L = layers.length;
    n = layers.clone();

    xi = new double[L][];
    h = new double[L][];
    theta = new double[L][];
    delta = new double[L][];
    d_theta = new double[L][];
    d_theta_prev = new double[L][];

    for (int lay = 0; lay < L; lay++) {
      xi[lay] = new double[n[lay]];
      h[lay] = new double[n[lay]];
      delta[lay] = new double[n[lay]];
      theta[lay] = new double[n[lay]];
      d_theta[lay] = new double[n[lay]];
      d_theta_prev[lay] = new double[n[lay]];

    }

    // Random initialization, but should have also negative values
    for (int lay = 0; lay < L; lay++) {
      for (int i = 0; i < n[lay]; i++) {
        theta[lay][i] = -1+2*(Math.random()); // new range [-1,1]
        d_theta[lay][i] = -1+2*(Math.random());
        d_theta_prev[lay][i] = -1+2*(Math.random());
      }
    }

    w = new double[L][][];
    d_w = new double[L][][];
    d_w_prev = new double[L][][];
    for (int lay = 1; lay < L; lay++) {
      w[lay] = new double[n[lay]][n[lay - 1]];
      d_w[lay] = new double[n[lay]][n[lay - 1]];
      d_w_prev[lay] = new double[n[lay]][n[lay - 1]];
    }

    // Random initialization, but should have also negative values
    for (int lay = 1; lay < L; lay++) {
      for (int i = 0; i < n[lay]; i++) {
        for (int j = 0; j < n[lay-1]; j++) {
          w[lay][i][j] = -1+2*(Math.random());
          d_w[lay][i][j] = -1+2*(Math.random());
          d_w_prev[lay][i][j] = -1+2*(Math.random());
        }
      }
    }

    fact = ActivationFunction.SIGMOID;

  }

  public double[] feedForward(double[] x_in) {

    double[] output = new double[n[L-1]];
    // copy input to the first layer (lay=0), Eq. (6)
    xi[0] = x_in;

    // feed-forward of input pattern
    // starting in layer 2 (lay=1)
    for (int lay = 1; lay < L; lay++) {
      // iteration i-units for each layer
      for (int i = 0; i < n[lay]; i++) {
        // calculate input field to unit i in layer lay, Eq. (8)
        // second term of the equation
        h[lay][i] = -theta[lay][i];
        // sum term of the equation
        for (int j = 0; j < n[lay - 1]; j++) {
          h[lay][i] = h[lay][i] + w[lay][i][j] * xi[lay - 1][j];
        }
        // Calculate the activation of layer lay, unit i, Eq. (7)
        xi[lay][i] = activationFunction(h[lay][i]);
      }
    }

    output = xi[L-1];
    // The output is just given by the activation of the units in the
    // output layer, Eq. (9)
    return (output);

  }

  public void WriteNNtoFile(String file_name) throws IOException {

    FileWriter fileWriter = new FileWriter(file_name);
    PrintWriter printWriter = new PrintWriter(fileWriter);

    printWriter.println("Length(L)");
    printWriter.printf(String.format(Locale.US,"%d\n",L));

    printWriter.println("NUnitsLayer(n)");
    for (int i = 0; i < n.length; i++) {
      printWriter.printf(String.format(Locale.US,"%d ",n[i]));
    }

    printWriter.println("\nFields(h)");
    for (int i = 0; i < h.length; i++) {
      for (int j = 0; j < h[i].length; j++) {
        printWriter.printf(String.format(Locale.US, "%f ", h[i][j]));
      }
      printWriter.println("");
    }

    printWriter.print("Activations(ξ)\n");
    for (int i = 0; i < xi.length; i++) {
      for (int j = 0; j < xi[i].length; j++) {
        printWriter.printf(String.format(Locale.US, "%f ", xi[i][j]));
      }
      printWriter.println("");
    }

    printWriter.print("Weights(w)\n");
    // start at first layer
    for (int i = 1; i < w.length; i++) {
      for (int j = 0; j < w[i].length; j++) {
        for (int k = 0; k < w[i][j].length; k++) {
          printWriter.printf(String.format(Locale.US, "%f ", w[i][j][k]));
        }
        printWriter.println("");
      }
    }

    printWriter.print("Thresholds(θ)\n");
    for (int i = 0; i < theta.length; i++) {
      for (int j = 0; j < theta[i].length; j++) {
        printWriter.printf(String.format(Locale.US, "%f ", theta[i][j]));
      }
      printWriter.println("");
    }

    printWriter.close();

  }
  public void errorBackPropagation(double[] y_out, double[] z){

    double sum;

    // compute the propagation error of the output layer (L-1), Eq. (11)
    for (int i = 0; i < n[L-1]; i++) {
      delta[L-1][i]=derivativeActivationFunction(h[L-1][i])*(y_out[i]-z[i]);
    }

    // Back-propagation of the errors to the rest of the network, Eq. (12)
    // (except the input layer)
    // starting in layer L (lay=L-1)
    for (int lay = L - 1; lay>0; lay--) {
      // iteration j-units for each layer
      for (int j = 0; j < n[lay-1]; j++) {
        // sum term of the equation
        sum = 0;
        for (int i = 0; i < n[lay]; i++) {
          sum = delta[lay][i]*w[lay][i][j] + sum;
        }
        delta[lay-1][j]=derivativeActivationFunction(h[lay-1][j])*sum;
      }
    }

  }
  public void updateWeightsThresholds(double learning_rate, double momentum){

    // starting at layer 1 (lay = 0)
    for (int lay = 0; lay < L; lay++) {
      // units of the layer lay
      for (int i = 0; i < n[lay]; i++) {
        if (lay > 0) {
          // i starting at layer 2 (lay 1) for the weights
          // j starting at layer 1 (lay 0): units of the lay-1
          for (int j = 0; j < n[lay - 1]; j++) {
            // Update the changes we applied in the previous step.
            d_w_prev[lay][i][j]=d_w[lay][i][j];
            // Calculate the modification of all weights, Eq. 14
            d_w[lay][i][j] = -learning_rate * delta[lay][i] * xi[lay - 1][j] + momentum * d_w_prev[lay][i][j];
            // Update all the weights, Eq. 15
            w[lay][i][j]=w[lay][i][j]+d_w[lay][i][j];
          }
        }
        // Update the changes we applied in the previous step.
        d_theta_prev[lay][i]=d_theta[lay][i];
        // Calculate the modification of all thresholds, Eq. 14
        d_theta[lay][i] = learning_rate * delta[lay][i] + momentum * d_theta_prev[lay][i];
        // Update all thresholds, Eq. 15
        theta[lay][i]=theta[lay][i]+d_theta[lay][i];
      }
    }

  }

  // Activation function - sigmoid, Eq. (10)
  public double activationFunction(double h) {

    double gh = 0;

    if (fact==ActivationFunction.SIGMOID)
      gh = 1 / (1 + Math.exp(-h));

    if (fact==ActivationFunction.RELU){
      if (h>=0)
        gh=h;
      else
        gh=0;
    }

    return(gh);
  }

  // Derivative of the activation function - sigmoid, Eq. (13)
  public double derivativeActivationFunction(double h) {

    double gh_der = 0;

    if (fact==ActivationFunction.SIGMOID)
      gh_der = activationFunction(h)*(1-activationFunction(h));
    if (fact==ActivationFunction.RELU) {
      if (h >= 0) // although never h == 0
        gh_der = 1;
      else
        gh_der = 0;
    }

    return(gh_der);
  }

}
