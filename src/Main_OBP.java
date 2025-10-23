import java.io.*;
import java.util.Locale;
import java.util.Scanner;

public class Main_OBP {

    public static void main(String[] args) throws IOException {

        // Uncomment the dataset to be used
        String name = "A1-turbine";
        //String name = "A1-synthetic";
        //String name = "A1-chemical";


        // IMPORT FROM TRAINING FILE
        String file_parameters = "../TRAINING_PARAMETERS/"+name+"-Training_parameters.txt";

        File file = new File(file_parameters);
        Scanner sc = new Scanner(file);

        String[] s;

        s = sc.nextLine().split(" ");
        String file_name_in = "../DATA_FILES/"+s[1];

        s = sc.nextLine().split(" ");
        int num_training_patterns = Integer.parseInt(s[1]);

        s = sc.nextLine().split(" ");
        int num_validation_patterns = Integer.parseInt(s[1]);

        s = sc.nextLine().split(" ");
        int num_test_patterns = Integer.parseInt(s[1]);

        s = sc.nextLine().split(" ");
        String activation_function = s[1];

        s = sc.nextLine().split(" ");
        int L = Integer.parseInt(s[1]);

        s = sc.nextLine().split(" ");
        int[] n = new int[L];
        for (int i = 1; i <= L; i++) {
            n[i-1] = Integer.parseInt(s[i]);
        }

        s = sc.nextLine().split(" ");
        int num_epoch = Integer.parseInt(s[1]);

        s = sc.nextLine().split(" ");
        double learning_rate = Double.parseDouble(s[1]);

        s = sc.nextLine().split(" ");
        double momentum = Double.parseDouble(s[1]);

        s = sc.nextLine().split(" ");
        String scaling_method = s[1];
        double s_min = Double.parseDouble(s[2]);
        double s_max = Double.parseDouble(s[3]);

        s = sc.nextLine().split(" ");
        String file_name_out_errors = "../DATA_RESULTS/"+name+"-"+s[1];

        s = sc.nextLine().split(" ");
        String file_name_out_validation = "../DATA_RESULTS/"+name+"-"+s[1];

        s = sc.nextLine().split(" ");
        String file_name_out_test = "../DATA_RESULTS/"+name+"-"+s[1];

        // Initialize the neural network, with all weights and thresholds randomly
        NeuralNet nn = new NeuralNet(n);

        // Read data from text file
        double[][] x_in = Functions.readFromFile(file_name_in).clone();

        // input pattern
        double[][] x_in_training = new double[num_training_patterns][];
        double[][] x_in_validation = new double[num_validation_patterns][];
        double[][] x_in_test = new double[num_test_patterns][];
        // output of the feed-forward propagation
        double[][] y_out_training = new double[num_training_patterns][];
        double[][] y_out_validation = new double[num_validation_patterns][];
        double[][] y_out_test = new double[num_test_patterns][];

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
            z_validation[pat] = new double[nn.n[nn.L-1]];
        }

        for (int pat = 0; pat < num_test_patterns; pat++) {
            x_in_test[pat] = new double[nn.n[0]];
            y_out_test[pat] = new double[nn.n[nn.L-1]];
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

        int length_cross_validation = 2;

        double[][] error_training = new double[length_cross_validation][num_epoch];
        double[][] error_validation = new double[length_cross_validation][num_epoch];


        // START

        // Cross validation of two sets

        for (int cr = 0; cr<length_cross_validation;cr++){

            for (int epoch = 0; epoch < num_epoch; epoch++) {

                for (int pat = 0; pat < num_training_patterns; pat++) {

                    // Choose a random pattern (x,z) of the training set
                    int random_pos = (int) (Math.random() * (num_training_patterns - 1));

                    // feed-forward propagation pattern x_in_training[random_pos] to obtain the output y_out[pat][0]
                    y_out_training[pat] = nn.feedForward(x_in_training[random_pos]).clone();

                    // Back-propagate the error for this pattern
                    nn.errorBackPropagation(y_out_training[pat], z_training[random_pos]);
                    // Update the weights and thresholds
                    nn.updateWeightsThresholds(learning_rate, momentum);
                }

                // Feed-forward all training patterns and calculate their prediction quadratic error

                for (int pat = 0; pat < num_training_patterns; pat++) {
                    y_out_training[pat] = nn.feedForward(x_in_training[pat]).clone();
                }
                error_training[cr][epoch] = Functions.predictionQuadraticError(y_out_training, z_training, num_training_patterns, nn.n[nn.L - 1]);

                // Feed-forward all validation patterns and calculate their prediction quadratic error
                for (int pat = 0; pat < num_validation_patterns; pat++) {
                    y_out_validation[pat] = nn.feedForward(x_in_validation[pat]).clone();
                }
                error_validation[cr][epoch] = Functions.predictionQuadraticError(y_out_validation, z_validation, num_validation_patterns, nn.n[nn.L - 1]);
            }

            if (cr==0){
                // Cross validation of two sets
                int j=0;
                for (int pat = num_validation_patterns; pat < num_training_patterns+num_validation_patterns; pat++) {
                    for (int i = 0; i < x_in[0].length-1; i++){
                        x_in_training[j][i] = x_in[pat][i];
                    }
                    z_training[j][0]=x_in[pat][x_in[0].length-1];
                    j++;
                }

                for (int pat = 0; pat < num_validation_patterns; pat++) {
                    for (int i = 0; i < x_in[0].length-1; i++){
                        x_in_validation[pat][i] = x_in[pat][i];
                    }
                    z_validation[pat][0]=x_in[pat][x_in[0].length-1];
                }
            }


        }

        // Save text file, plot the data and descale in Matlab

        // Feed-forward all test patterns
        for (int pat = 0; pat < num_test_patterns; pat++) {
            y_out_test[pat] = nn.feedForward(x_in_test[pat]).clone();
        }

        // Errors
        String title = "Error-training Error-validation";
        double[] err_training = new double [num_epoch];
        double[] err_validation = new double [num_epoch];

        for (int i = 0; i < num_epoch; i++){
            err_training[i] = (error_training[0][i]+error_training[1][i])/2;
            err_validation[i] = (error_validation[0][i]+error_validation[1][i])/2;
        }
        Functions.writeToFile(file_name_out_errors, err_training, err_validation, title);

        // Validation of last epoch and second iteration set of crossvalidation
        double[] y_out_validation_file = new double[y_out_validation.length];
        double[] z_validation_file = new double[z_validation.length];
        for (int i = 0; i<y_out_validation.length;i++) {
            y_out_validation_file[i] = y_out_validation[i][0];
            z_validation_file[i]=z_validation[i][0];
        }
        title = "y_o_u_tvalidation z_o_u_tvalidation";
        Functions.writeToFile(file_name_out_validation, y_out_validation_file, z_validation_file,title);

        // Test
        double[] y_out_test_file = new double[y_out_test.length];
        double[] z_test_file = new double[z_test.length];
        for (int i = 0; i<y_out_test.length;i++) {
            y_out_test_file[i] = y_out_test[i][0];
            z_test_file[i]=z_test[i][0];
        }
        title = "y_o_u_ttest z_o_u_ttest";
        Functions.writeToFile(file_name_out_test, y_out_test_file, z_test_file,title);

        // Save the best parameters to get a trained neural-network (feed-forward):
        nn.WriteNNtoFile("../TRAINED_NN/"+name+"-nn.txt");


    }


}
