
// 9) Smith Number : A  Smith number is a composite number, the sum of whose digits is the sum of the digits of its prime factors obtained as a result of prime factorisation (excluding 1). The first few such numbers are 4, 22, 27, 58, 85, 94, 121 ………………..

public class Smith {
    public static void main(String[] args) {
        int num = 85; // Example number
        int originalNum = num;
        int sumOfDigits = sumOfDigits(num);
        int sumOfPrimeFactorDigits = sumOfPrimeFactorDigits(num);

        // Check if the sum of digits is equal to the sum of prime factor digits
        if (sumOfDigits == sumOfPrimeFactorDigits && !isPrime(originalNum)) {
            System.out.println(originalNum + " is a Smith number.");
        } else {
            System.out.println(originalNum + " is not a Smith number.");
        }
    }

    // Helper method to calculate the sum of digits of a number
    private static int sumOfDigits(int n) {
        int sum = 0;
        while (n != 0) {
            sum += n % 10;
            n /= 10;
        }
        return sum;
    }

    // Helper method to calculate the sum of digits of prime factors
    private static int sumOfPrimeFactorDigits(int n) {
        int sum = 0;
        for (int i = 2; i <= n; i++) {
            while (n % i == 0) {
                sum += sumOfDigits(i);
                n /= i;
            }
        }
        return sum;
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
