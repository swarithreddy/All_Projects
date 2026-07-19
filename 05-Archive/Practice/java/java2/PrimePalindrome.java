// 11) Prime Palindrome : A prime palindrome integer is a positive integer (without leading zeros) which is prime as well as a palindrome.
public class PrimePalindrome {
    public static void main(String[] args) {
        int num = 131; // Example number
        int originalNum = num;

        // Check if the number is a palindrome
        if (isPalindrome(num) && isPrime(num)) {
            System.out.println(originalNum + " is a Prime Palindrome.");
        } else {
            System.out.println(originalNum + " is not a Prime Palindrome.");
        }
    }

    // Helper method to check if a number is a palindrome
    private static boolean isPalindrome(int n) {
        int reversedNum = 0;
        int originalNum = n;

        while (n != 0) {
            int digit = n % 10;
            reversedNum = reversedNum * 10 + digit;
            n /= 10;
        }

        return originalNum == reversedNum;
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
