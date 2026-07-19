// 10) Armstrong Number : An Armstrong number of three digits is an integer such that the sum of the cubes of its digits is equal to the number itself.
public class Armstrong {
    public static void main(String[] args) {
        int num = 153; // Example number
        int originalNum = num;
        int sumOfCubes = 0;

        // Calculate the sum of cubes of each digit
        while (num != 0) {
            int digit = num % 10;
            sumOfCubes += digit * digit * digit;
            num /= 10;
        }

        // Check if the sum of cubes is equal to the original number
        if (sumOfCubes == originalNum) {
            System.out.println(originalNum + " is an Armstrong number.");
        } else {
            System.out.println(originalNum + " is not an Armstrong number.");
        }
    }
}
