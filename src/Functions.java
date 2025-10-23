import java.io.*;
import java.util.Locale;
import java.util.Scanner;

public class Functions {

    public static double predictionQuadraticError(double[][] y_out, double[][] z, int p, int m){

        double error = 0;

        for (int j = 0; j < p; j++) {
            for (int i=0; i<m;i++) {
                error = error + Math.pow((y_out[j][i]-z[j][i]),2);
            }
        }

        error = 0.5 * error;
        return error;
    }
    public static double[][] readFromFile(String file_name) throws FileNotFoundException {
        File file = new File(file_name);
        Scanner sc = new Scanner(file);

        String[] s;
        int j = 0;

        double[][] values = new double[2000][sc.nextLine().split(" ").length];

        while (sc.hasNextLine()) {
            s = sc.nextLine().split(" ");
            for (int i = 0; i< values[0].length; i++) {
                values[j][i]=Double.parseDouble(s[i]);
            }
            j=j+1;
        }

        double[][] valuesToReturn = new double[j][values[0].length];

        for (int i = 0; i<valuesToReturn.length; i++) {
            for (int k = 0; k < valuesToReturn[0].length; k++)
                valuesToReturn[i][k] = values[i][k];
        }

        return valuesToReturn;

    }

    public static void writeParametersToFile(String file_name, String file_name_in, int num_training_patterns, int num_validation_patterns, int num_test_patterns, ActivationFunction fact, int L, int[] n, double num_epoch, double learning_rate, double momentum, String scaling_method, double s_min, double s_max, String output_file1, String output_file2, String output_file3) throws IOException {

        FileWriter fileWriter = new FileWriter(file_name);
        PrintWriter printWriter = new PrintWriter(fileWriter);

        printWriter.print("data_file: ");
        printWriter.printf(file_name_in);

        printWriter.print("\nnum_training_patterns: ");
        printWriter.printf("%d \n", num_training_patterns);

        printWriter.print("num_validation_patterns: ");
        printWriter.printf("%d \n", num_validation_patterns);

        printWriter.print("num_test_patterns: ");
        printWriter.printf(Locale.US, "%d \n", num_test_patterns);

        printWriter.print("activation_function: ");
        printWriter.printf(fact.toString());

        printWriter.print("\nL= ");
        printWriter.printf("%d \n", L);

        printWriter.print("n= ");
        for (int i = 0; i < n.length; i++) {
            printWriter.printf( "%d ", n[i]);
        }

        printWriter.print("\nn_epoch= ");
        printWriter.printf("%d \n", (int)num_epoch);

        printWriter.print("learning_rate= ");
        printWriter.printf(String.format(Locale.US, "%f \n", learning_rate));

        printWriter.print("momentum= ");
        printWriter.printf(String.format(Locale.US, "%f \n", momentum));

        printWriter.print("scaling_method= ");
        printWriter.print(scaling_method);
        printWriter.printf(String.format(Locale.US," %.2f %f\n", s_min, s_max));

        printWriter.print("output_file1= ");
        printWriter.printf(output_file1);

        printWriter.print("\noutput_file2= ");
        printWriter.printf(output_file2);

        printWriter.print("\noutput_file3= ");
        printWriter.printf(output_file3);

        printWriter.close();


    }

    public static void writeToFile(String file_name, double[] out1, double[] out2, String title) throws IOException {
        FileWriter fileWriter = new FileWriter(file_name);
        PrintWriter printWriter = new PrintWriter(fileWriter);

        printWriter.println(title);
        for (int i = 0; i < out1.length; i++) {
            printWriter.printf(String.format(Locale.US,"%f %f \n",out1[i], out2[i]));
        }
        printWriter.close();
    }


}
