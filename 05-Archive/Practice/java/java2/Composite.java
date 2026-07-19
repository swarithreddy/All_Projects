// 15) Composite Number : A number is said to be a composite, if it
// has one or more then one factors excluding 1 and the number itself.
public class Composite {
    public static void main(String[] args) {
        int num = 12; // Example number
        if (isComposite(num)) {
            System.out.println(num + " is a Composite number.");
        } else {
            System.out.println(num + " is not a Composite number.");
        }
    }

    // Helper method to check if a number is composite
    private static boolean isComposite(int n) {
        if (n <= 1) return false; // Numbers less than or equal to 1 are not composite
        int factorCount = 0;
        for (int i = 2; i <= Math.sqrt(n); i++) {
            if (n % i == 0) {
                factorCount++;
                if (i != n / i) { // Count the complementary factor only if it's different
                    factorCount++;
                }
            }
            if (factorCount > 1) {
                return true; // More than one factor found
            }
        }
        return false; // No factors found, hence not composite
    }
}
