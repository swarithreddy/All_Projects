// 16) Circular Prime Number : A circular prime is a prime number with the property that the number generated at each intermediate step when cyclically permuting its (base 10) digits will be prime.For example, 1193 is a circular prime, since 1931, 9311 and 3119 all are also prime. A circular prime with at least two digits can only consist of combinations of the digits 1, 3, 7 or 9, because having 0, 2, 4, 6 or 8 as the last digit makes the number divisible by 2, and having 0 or 5 as the last digit makes it divisible by 5. The complete listing of the smallest representative prime from all known cycles of circular primes  is 2, 3, 5, 7, 13, 17, 37, 79, 113, 197, 199, 337, 1193, 3779, 11939, 19937, 193939, 199933.

public class CircularPrime {
    public static void main(String[] args) {
        int num = 197; // Example number
        int originalNum = num;

        if (isCircularPrime(num)) {
            System.out.println(originalNum + " is a Circular Prime.");
        } else {
            System.out.println(originalNum + " is not a Circular Prime.");
        }
    }

    // Helper method to check if a number is a circular prime
    private static boolean isCircularPrime(int n) {
        String numStr = Integer.toString(n);
        int len = numStr.length();

        for (int i = 0; i < len; i++) {
            if (!isPrime(n)) {
                return false;
            }
            // Rotate the number
            n = rotateNumber(n, len);
        }
        return true;
    }

    // Helper method to rotate the number
    private static int rotateNumber(int n, int len) {
        String numStr = Integer.toString(n);
        String rotatedStr = numStr.substring(1) + numStr.charAt(0);
        return Integer.parseInt(rotatedStr);
    }

    // Helper method to check if a number is prime
    private static boolean isPrime(int n) {
        if (n <= 1) return false;
        for (int i = 2; i <= Math.sqrt(n); i++) {
            if (n % i == 0) return false;
        }
        return true;
    }    
}
