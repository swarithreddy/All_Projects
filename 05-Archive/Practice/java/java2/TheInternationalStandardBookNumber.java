// 14) The International Standard Book Number (ISBN) : is a unique numeric book identifier which is printed on every book. The ISBN is based upon a 10-digit code. The ISBN is legal if:
// 1xdigit1 + 2xdigit2 + 3xdigit3 + 4xdigit4 + 5xdigit5 + 6xdigit6 + 7xdigit7 + 8xdigit8 + 9xdigit9 + 10xdigit10 is divisible by 11.
public class TheInternationalStandardBookNumber{
    public static void main(String[] args) {
        String isbn = "0306406152"; // Example ISBN number
        if (isValidISBN(isbn)) {
            System.out.println(isbn + " is a valid ISBN number.");
        } else {
            System.out.println(isbn + " is not a valid ISBN number.");
        }
    }

    public static boolean isValidISBN(String isbn) {
        if (isbn.length() != 10) {
            return false; // ISBN must be 10 digits
        }

        int sum = 0;
        for (int i = 0; i < 10; i++) {
            char c = isbn.charAt(i);
            if (!Character.isDigit(c)) {
                return false; // Invalid character found
            }
            int digit = Character.getNumericValue(c);
            sum += (i + 1) * digit;
        }

        return sum % 11 == 0; // Check divisibility by 11
    }
}
