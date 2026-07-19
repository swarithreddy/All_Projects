// Disarum Number : A number will be called DISARIUM if sum of its digits powered with their respective position is equal to the original number.
public class Disarum {
    public static void main(String[] args) {
        int num = 135; // Example number
        if (isDisarum(num)) {
            System.out.println(num + " is a Disarum number.");
        } else {
            System.out.println(num + " is not a Disarum number.");
        }
    }

    public static boolean isDisarum(int number) {
        String numStr = Integer.toString(number);
        int sum = 0;

        // Calculate the sum of digits powered with their respective positions
        for (int i = 0; i < numStr.length(); i++) {
            int digit = Character.getNumericValue(numStr.charAt(i));
            sum += Math.pow(digit, i + 1);
        }

        // Check if the sum is equal to the original number
        return sum == number;
    }
}
