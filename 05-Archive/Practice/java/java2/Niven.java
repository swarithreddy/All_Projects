public class Niven {
    public static void main(String[] args) {
        int num = 18; // Example number
        if (isNiven(num)) {
            System.out.println(num + " is a Niven (Harshad) number.");
        } else {
            System.out.println(num + " is not a Niven (Harshad) number.");
        }
    }

    public static boolean isNiven(int number) {
        int sumOfDigits = 0;
        int temp = number;

        // Calculate the sum of digits
        while (temp != 0) {
            sumOfDigits += temp % 10;
            temp /= 10;
        }

        // Check if the number is divisible by the sum of its digits
        return number % sumOfDigits == 0;
    }
}
