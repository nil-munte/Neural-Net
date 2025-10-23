import java.io.File;
import java.io.IOException;
import java.util.Scanner;

public class Main_FeedForwardTest {

    public static void main(String[] args) throws IOException {

        // Uncomment the dataset to be used
        //String name = "A1-turbine";
        //String name = "A1-synthetic";
        String name = "A1-chemical";

        String file_name_in = "../DATA_FILES/"+name+"-scaled.txt";
        String file_nn_name_in = "../TRAINED_NN/"+name+"-nn.txt";
        String file_name_out_test = "../DATA_RESULTS/"+name+"-Test-feedforward.txt";

        // IMPORT NN FROM FILE

        File file = new File(file_nn_name_in);
        Scanner sc = new Scanner(file);

        String[] s;

        sc.nextLine();
        int L = Integer.parseInt(sc.nextLine());

        sc.nextLine();
        int[] layers = new int[L];
        s = sc.nextLine().split(" ");
        for (int i = 0; i<L; i++) {
            layers[i]=Integer.parseInt(s[i]);
        }

        NeuralNet nn = new NeuralNet(layers);

        sc.nextLine();
        for (int lay = 0; lay < L; lay++) {
            s = sc.nextLine().split(" ");
            for (int i = 0; i < nn.n[lay]; i++) {
                nn.h[lay][i]=Double.parseDouble(s[i]);
            }
        }

        sc.nextLine();
        for (int lay = 0; lay < L; lay++) {
            s = sc.nextLine().split(" ");
            for (int i = 0; i < nn.n[lay]; i++) {
                nn.xi[lay][i]=Double.parseDouble(s[i]);
            }
        }

        sc.nextLine();
        for (int lay = 1; lay < L; lay++) {
            for (int i = 0; i < nn.n[lay]; i++) {
                s = sc.nextLine().split(" ");
                for (int j = 0; j < nn.n[lay-1]; j++) {
                    nn.w[lay][i][j]=Double.parseDouble(s[j]);
                }
            }
        }

        sc.nextLine();
        for (int lay = 0; lay < L; lay++) {
            s = sc.nextLine().split(" ");
            for (int i = 0; i < nn.n[lay]; i++) {
                nn.theta[lay][i]=Double.parseDouble(s[i]);
            }
        }

        // IMPORT TRAINING PATTERNS

        int num_training_patterns = 300;
        int num_validation_patterns = 83;
        int num_test_patterns = 68;


        // Read data from text file
        double[][] x_in = Functions.readFromFile(file_name_in).clone();

        // input pattern
        double[][] x_in_test = new double[num_test_patterns][];

        // output of the feed-forward propagation
        double[][] y_out_test = new double[num_test_patterns][];

        // desired output
        double[][] z_test = new double[num_test_patterns][];

        for (int pat = 0; pat < num_test_patterns; pat++) {
            x_in_test[pat] = new double[nn.n[0]];
            y_out_test[pat] = new double[nn.n[nn.L-1]];
            z_test[pat] = new double[nn.n[nn.L-1]];
        }

        for (int pat = 0; pat < num_test_patterns; pat++) {
            for (int i = 0; i < x_in[0].length-1; i++){
                x_in_test[pat][i] = x_in[pat+num_training_patterns+num_validation_patterns][i];
            }
            z_test[pat][0]=x_in[pat+num_training_patterns+num_validation_patterns][x_in[0].length-1];
        }

        // Feed-forward all test patterns
        for (int pat = 0; pat < num_test_patterns; pat++) {
            y_out_test[pat] = nn.feedForward(x_in_test[pat]).clone();
        }

        double[] y_out_test_file = new double[y_out_test.length];
        double[] z_test_file = new double[z_test.length];
        for (int i = 0; i<y_out_test.length;i++) {
            y_out_test_file[i] = y_out_test[i][0];
            z_test_file[i]=z_test[i][0];
        }

        String title = "y_o_u_ttest z_o_u_ttest";
        Functions.writeToFile(file_name_out_test, y_out_test_file, z_test_file,title);


    }


}
