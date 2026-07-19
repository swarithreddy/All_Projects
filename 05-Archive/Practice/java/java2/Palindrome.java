// 6) Palindrome Number : A number is said to be a palindrome number when is reverse is equal to the number itself. In other words, it is the same if you read it from left to right or vice-versa OR it is a number that remains the same when its digits are reversed. Like 16461, for example, it is "symmetrical".
public class Palindrome {
    public static void main(String[] args) {
        int num = 16461; // Example number
        int originalNum = num;
        int reversedNum = 0;

        // Reverse the number
        while (num != 0) {
            int digit = num % 10;
            reversedNum = reversedNum * 10 + digit;
            num /= 10;
        }

        // Check if the original number is equal to the reversed number
        if (originalNum == reversedNum) {
            System.out.println(originalNum + " is a Palindrome number.");
        } else {
            System.out.println(originalNum + " is not a Palindrome number.");
        }
    }
}