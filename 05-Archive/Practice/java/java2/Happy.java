// 17) Happy Number : If a number is happy, then all members of its sequence are happy; if a number is unhappy, all members of the sequence are unhappy.
public class Happy {
    public static void main(String[] args) {
        int num = 19; // Example number
        if (isHappy(num)) {
            System.out.println(num + " is a Happy number.");
        } else {
            System.out.println(num + " is not a Happy number.");
        }
    }

    // Method to determine if a number is happy
    private static boolean isHappy(int n) {
        int slow = n;
        int fast = n;

        do {
            slow = sumOfSquares(slow); // Move one step
            fast = sumOfSquares(sumOfSquares(fast)); // Move two steps
        } while (slow != fast);

        return slow == 1; // If they meet at 1, it's a happy number
    }

    // Helper method to calculate the sum of the squares of the digits
    private static int sumOfSquares(int n) {
        int sum = 0;
        while (n != 0) {
            int digit = n % 10;
            sum += digit * digit;
            n /= 10;
        }
        return sum;
    }
}
