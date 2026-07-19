// 12) Fibonacci series Number : The number belonging to Fibonacci Series.
//     Fibonacci series is the series in which except first two integers, the sum of       the two consecutive numbers is the third number.
public class FibonacciseriesNumber {
    public static void main(String[] args) {
        int n = 10; // Number of terms in the Fibonacci series
        int first = 0, second = 1;

        System.out.println("Fibonacci Series up to " + n + " terms:");

        for (int i = 1; i <= n; i++) {
            System.out.print(first + " ");
            int next = first + second;
            first = second;
            second = next;
        }
    }
}
