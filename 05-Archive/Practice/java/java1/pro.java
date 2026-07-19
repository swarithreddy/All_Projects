import java.io.*;

public class pro {
    public static void main(String[] args) {
        String source = "input.txt";      // Source file path
        String destination = "output.txt"; // Destination file path

        try (
            BufferedReader reader = new BufferedReader(new FileReader(source));
            BufferedWriter writer = new BufferedWriter(new FileWriter(destination))
        ) {
            String line;
            while ((line = reader.readLine()) != null) {
                writer.write(line);
                writer.newLine(); // Write a newline character
            }
            System.out.println("File copied successfully.");
        } catch (IOException e) {
            System.out.println("An error occurred: " + e.getMessage());
        }
    }
}
