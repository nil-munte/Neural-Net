import java.io.*;
import java.util.Locale;
import java.util.Scanner;

public class Main_Parameters {

    public static void main(String[] args) throws IOException {

        // Uncomment the dataset to be used
        //String name = "A1-turbine";
        //String name = "A1-synthetic";
        String name = "A1-chemical";

        // IMPORT FROM TRAINING FILE
        String file_parameters = "../TRAINING_PARAMETERS/"+name+"-Possible-Training_parameters.txt";

        File file = new File(file_parameters);
        Scanner sc = new Scanner(file);

        String[] s;

        s = sc.nextLine().split(" ");
        int num_training_patterns = Integer.parseInt(s[1]);

        s = sc.nextLine().split(" ");
        int num_validation_patterns = Integer.parseInt(s[1]);

        s = sc.nextLine().split(" ");
        int num_test_patterns = Integer.parseInt(s[1]);

        s = sc.nextLine().split(" ");
        double[] learning_rate = new double[s.length-1];
        for (int i = 1; i < s.length; i++) {
            learning_rate[i-1] = Double.parseDouble(s[i]);
        }

        // Learning rate: can be chosen from the following loop (more values) (uncomment), or from the text file

        // Learning rate from 0.01 to 0.2 in steps 0.01
        /*double[] learning_rate = new double [20];
        learning_rate[0]=0.01;
        int learn = 1;
        while (learning_rate[learn-1] <= 0.2){
          learning_rate[learn]=learning_rate[learn-1]+0.01;
          learn++;
        }*/

        s = sc.nextLine().split(" ");
        double[] momentum = new double[s.length-1];
        for (int i = 1; i < s.length; i++) {
            momentum[i-1] = Double.parseDouble(s[i]);
        }

        s = sc.nextLine().split(" ");
        int[] num_epoch = new int[s.length-1];
        for (int i = 1; i < s.length; i++) {
            num_epoch[i-1] = Integer.parseInt(s[i]);
        }

        s = sc.nextLine().split(" ");
        int layer_combinations = Integer.parseInt(s[1]);

        int[][] layers = new int[layer_combinations][];
        int layer = 0;
        while (sc.hasNextLine()){
            s = sc.nextLine().split(" ");
            layers[layer] = new int[s.length];
            for (int i = 0; i < s.length; i++) {
                layers[layer][i] = Integer.parseInt(s[i]);
            }
            layer++;
        }

        double[] error_training;
        double[] error_training_best = new double[0]; // just to initialize

        double[] error_validation;
        double[] error_validation_best = new double[0]; // just to initialize

        double min_error = 0;

        double[] parameters = {learning_rate[0],momentum[0],num_epoch[0]};
        int[] parameters_layers = layers[0];

        // Initialize the neural network, with all weights and thresholds randomly
        NeuralNet nn = new NeuralNet(layers[0]);

        String file_name = "../TRAINING_PARAMETERS/"+name+"-Training_parameters.txt";
        String file_name_in = "../DATA_FILES/"+name+"-scaled.txt";
        String file_name_out_errors = "../DATA_RESULTS/"+name+"-Errors-Parameters.txt";
        String file_name_out_validation= "../DATA_RESULTS/"+name+"-Validation-Parameters.txt";
        String file_name_out_test = "../DATA_RESULTS/"+name+"-Test-Parameters.txt";

        // Read data from text file
        double[][] x_in = Functions.readFromFile(file_name_in).clone();

        // input pattern
        double[][] x_in_training = new double[num_training_patterns][];
        double[][] x_in_validation = new double[num_validation_patterns][];
        double[][] x_in_test = new double[num_test_patterns][];

        // output of the feed-forward propagation
        double[][] y_out_training = new double[num_training_patterns][];

        double[][] y_out_validation = new double[num_validation_patterns][];
        double[][] y_out_validation_best = new double[num_validation_patterns][];

        double[][] y_out_test = new double[num_test_patterns][];
        double[][] y_out_test_best = new double[num_test_patterns][];

        // desired output
        double[][] z_training = new double[num_training_patterns][];

        double[][] z_validation = new double[num_validation_patterns][];

        double[][] z_test = new double[num_test_patterns][];

        for (int pat = 0; pat < num_training_patterns; pat++) {
            x_in_training[pat] = new double[nn.n[0]];
            y_out_training[pat] = new double[nn.n[nn.L-1]];
            z_training[pat] = new double[nn.n[nn.L-1]];
        }

        for (int pat = 0; pat < num_validation_patterns; pat++) {
            x_in_validation[pat] = new double[nn.n[0]];
            y_out_validation[pat] = new double[nn.n[nn.L-1]];
            y_out_validation_best[pat] = new double[nn.n[nn.L-1]];
            z_validation[pat] = new double[nn.n[nn.L-1]];
        }

        for (int pat = 0; pat < num_test_patterns; pat++) {
            x_in_test[pat] = new double[nn.n[0]];
            y_out_test[pat] = new double[nn.n[nn.L-1]];
            y_out_test_best[pat] = new double[nn.n[nn.L-1]];
            z_test[pat] = new double[nn.n[nn.L-1]];
        }

        for (int pat = 0; pat < num_training_patterns; pat++) {
            for (int i = 0; i < x_in[0].length-1; i++){
                x_in_training[pat][i] = x_in[pat][i];
            }
            z_training[pat][0]=x_in[pat][x_in[0].length-1];
        }

        for (int pat = 0; pat < num_validation_patterns; pat++) {
            for (int i = 0; i < x_in[0].length-1; i++){
                x_in_validation[pat][i] = x_in[pat+num_training_patterns][i];
            }
            z_validation[pat][0]=x_in[pat+num_training_patterns][x_in[0].length-1];
        }

        for (int pat = 0; pat < num_test_patterns; pat++) {
            for (int i = 0; i < x_in[0].length-1; i++){
                x_in_test[pat][i] = x_in[pat+num_training_patterns+num_validation_patterns][i];
            }
            z_test[pat][0]=x_in[pat+num_training_patterns+num_validation_patterns][x_in[0].length-1];
        }

        // START
        for (int lay_config = 0; lay_config < layers.length ; lay_config++) {
            for (int i = 0; i<learning_rate.length;i++){
                for (int j = 0; j<momentum.length;j++) {
                    for (int k = 0; k < num_epoch.length; k++) {

                        error_training = new double[num_epoch[k]];
                        error_validation = new double[num_epoch[k]];

                        // Reinitialize the NN
                        nn = new NeuralNet(layers[lay_config]);

                        for (int epoch = 0; epoch < num_epoch[k]; epoch++) {

                            for (int pat = 0; pat < num_training_patterns; pat++) {

                                // Choose a random pattern (x,z) of the training set
                                int random_pos = (int) (Math.random() * (num_training_patterns - 1));

                                // feed-forward propagation pattern x_in_training[random_pos] to obtain the output y_out[pat][0]
                                y_out_training[pat] = nn.feedForward(x_in_training[random_pos]).clone();

                                // Back-propagate the error for this pattern
                                nn.errorBackPropagation(y_out_training[pat], z_training[random_pos]);
                                // Update the weights and thresholds
                                nn.updateWeightsThresholds(learning_rate[i], momentum[j]);
                            }

                            // Feed-forward all training patterns and calculate their prediction quadratic error

                            for (int pat = 0; pat < num_training_patterns; pat++) {
                                y_out_training[pat] = nn.feedForward(x_in_training[pat]).clone();
                            }
                            error_training[epoch] = Functions.predictionQuadraticError(y_out_training, z_training, num_training_patterns, nn.n[nn.L - 1]);

                            // Feed-forward all validation patterns and calculate their prediction quadratic error
                            for (int pat = 0; pat < num_validation_patterns; pat++) {
                                y_out_validation[pat] = nn.feedForward(x_in_validation[pat]).clone();
                            }
                            error_validation[epoch] = Functions.predictionQuadraticError(y_out_validation, z_validation, num_validation_patterns, nn.n[nn.L - 1]);

                        }

                        // Feed-forward all test patterns
                        for (int pat = 0; pat < num_test_patterns; pat++) {
                            y_out_test[pat] = nn.feedForward(x_in_test[pat]).clone();
                        }

                        if (i==0 && j==0 && k==0 && lay_config==0){
                            min_error = error_validation[0];
                        }
                        else if(min_error > error_validation[num_epoch[k]-1]) {
                            min_error = error_validation[num_epoch[k]-1];
                            parameters[0]=learning_rate[i];
                            parameters[1]=momentum[j];
                            parameters[2]=num_epoch[k];
                            parameters_layers = layers[lay_config];

                            error_training_best=error_training.clone();
                            error_validation_best=error_validation.clone();
                            y_out_validation_best=y_out_validation.clone();
                            y_out_test_best=y_out_test.clone();

                        }

                    }
                }
            }
        }

        System.out.println("Learning rate: "+parameters[0]+"\nMomentum: "+parameters[1]+"\nNum_Epoch: "+parameters[2]);
        System.out.print("Layers: ");
        for (int i = 0; i<parameters_layers.length;i++)
            System.out.print(parameters_layers[i]+" ");

        // Save the best parameters, errors and best predicted values obtained in a text file
        Functions.writeParametersToFile(file_name,name+"-scaled.txt", num_training_patterns, num_validation_patterns, num_test_patterns, nn.fact, parameters_layers.length, parameters_layers, parameters[2], parameters[0], parameters[1], "normalization", 0.1, 0.9, "Errors.txt", "Validation.txt", "Test.txt");

        // Plot the data and descale in Matlab for the best performing BP.

        // Errors
        String title = "Error-training Error-validation";
        Functions.writeToFile(file_name_out_errors, error_training_best, error_validation_best, title);

        // Validation
        double[] y_out_validation_file = new double[y_out_validation_best.length];
        double[] z_validation_file = new double[z_validation.length];
        for (int i = 0; i<y_out_validation_best.length;i++) {
            y_out_validation_file[i] = y_out_validation_best[i][0];
            z_validation_file[i]=z_validation[i][0];
        }
        title = "y_o_u_tvalidation z_o_u_tvalidation";
        Functions.writeToFile(file_name_out_validation, y_out_validation_file, z_validation_file,title);

        // Test
        double[] y_out_test_file = new double[y_out_test_best.length];
        double[] z_test_file = new double[z_test.length];
        for (int i = 0; i<y_out_test_best.length;i++) {
            y_out_test_file[i] = y_out_test_best[i][0];
            z_test_file[i]=z_test[i][0];
        }
        title = "y_o_u_ttest z_o_u_ttest";
        Functions.writeToFile(file_name_out_test, y_out_test_file, z_test_file,title);
    }


}
