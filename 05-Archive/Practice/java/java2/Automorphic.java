// 5) Automorphic Number : An automorphic number is a number which is present in the last digit(s) of its square.
public class Automorphic {
    public static void main(String[] args) {
        int num = 76; // Example number
        int square = num * num;

        // Convert both numbers to strings
        String numStr = Integer.toString(num);
        String squareStr = Integer.toString(square);

        // Check if the square ends with the number
        if (squareStr.endsWith(numStr)) {
            System.out.println(num + " is an Automorphic number.");
        } else {
            System.out.println(num + " is not an Automorphic number.");
        }
    }
}
