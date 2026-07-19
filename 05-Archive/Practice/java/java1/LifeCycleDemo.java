import java.applet.Applet;
import java.awt.Graphics;

/*
<applet code="LifeCycleDemo.class" width="400" height="300">
    <param name="user" value="Swarith">
    <param name="greeting" value="Welcome to Applet Programming!">
</applet>
*/

public class LifeCycleDemo extends Applet {
    String userName;
    String greetMessage;

    // Called once when the applet is first loaded
    public void init() {
        userName = getParameter("user");       // Getting parameter 'user'
        greetMessage = getParameter("greeting"); // Getting parameter 'greeting'
        if (userName == null) userName = "Guest";
        if (greetMessage == null) greetMessage = "Hello!";
    }

    // Called when the applet becomes active
    public void start() {
        System.out.println("Applet Started");
    }

    // Called when the applet needs to repaint itself
    public void paint(Graphics g) {
        g.drawString(greetMessage, 50, 100);
        g.drawString("User: " + userName, 50, 130);
    }

    // Called when the applet is stopped or inactive
    public void stop() {
        System.out.println("Applet Stopped");
    }

    // Called when the applet is about to be destroyed
    public void destroy() {
        System.out.println("Applet Destroyed");
    }
}
