// 7) Special Number : A number is said to be special number when the sum of factorial of its digits is equal to the number itself.
public class Special {
    public static void main(String[] args) {
        int num = 145; // Example number
        int originalNum = num;
        int sumOfFactorials = 0;

        // Calculate the sum of factorials of each digit
        while (num != 0) {
            int digit = num % 10;
            sumOfFactorials += factorial(digit);
            num /= 10;
        }

        // Check if the sum of factorials is equal to the original number
        if (sumOfFactorials == originalNum) {
            System.out.println(originalNum + " is a Special number.");
        } else {
            System.out.println(originalNum + " is not a Special number.");
        }
    }

    // Helper method to calculate factorial of a number
    private static int factorial(int n) {
        if (n == 0 || n == 1) {
            return 1;
        }
        int fact = 1;
        for (int i = 2; i <= n; i++) {
            fact *= i;
        }
        return fact;
    }
}
