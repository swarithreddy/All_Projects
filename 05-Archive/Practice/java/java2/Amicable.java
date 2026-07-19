// 13) Amicable Number : If two numbers are such that the sum of the perfect divisors of one number is equal to the other number and the sum of the perfect divisors of  the other number is equal to the first number, then the numbers are called Amicable Numbers.
public class Amicable {
    public static void main(String[] args) {
        int num1 = 220; // First example number
        int num2 = 284; // Second example number

        int sumOfDivisorsNum1 = sumOfDivisors(num1);
        int sumOfDivisorsNum2 = sumOfDivisors(num2);

        // Check if the numbers are amicable
        if (sumOfDivisorsNum1 == num2 && sumOfDivisorsNum2 == num1) {
            System.out.println(num1 + " and " + num2 + " are Amicable numbers.");
        } else {
            System.out.println(num1 + " and " + num2 + " are not Amicable numbers.");
        }
    }

    // Helper method to calculate the sum of proper divisors of a number
    private static int sumOfDivisors(int n) {
        int sum = 0;
        for (int i = 1; i <= n / 2; i++) {
            if (n % i == 0) {
                sum += i;
            }
        }
        return sum;
    }
}
