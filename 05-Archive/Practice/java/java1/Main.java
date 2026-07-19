
class one extends Thread {

    public void run() {
        for (int i = 0; i < 10; i++) {
            try {
                System.out.println("Thread one is running");
                Thread.sleep(1000);
            } catch (Exception e) {
                System.out.println("Exception in thread one: " + e.getMessage());
            }
        }
    }
}

class two extends Thread {

    public void run() {
        for (int i = 0; i < 10; i++) {
            try {
                Thread.sleep(1000);
                System.out.println("Thread two is running");// Simulating some work
            } catch (InterruptedException e) {
                System.out.println("Exception in thread two: " + e.getMessage());
            }
        }
    }
}

public class Main {

    public static void main(String[] args) {
        one t = new one();
        t.start();
        two t1 = new two();
        t1.start();
    }

}
