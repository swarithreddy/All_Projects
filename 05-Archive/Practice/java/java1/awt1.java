import java.awt.*;
import java.awt.event.*;

public class awt1 extends Frame implements MouseListener {
    Label l;

    awt1() {
        l = new Label("Click anywhere");
        l.setBounds(20, 50, 200, 20);
        add(l);
        
        setSize(300, 300);
        setLayout(null);
        setVisible(true);
        addMouseListener(this);
    }

    public void mouseClicked(MouseEvent e) {
        l.setText("Mouse Clicked at: " + e.getX() + ", " + e.getY());
    }

    public void mousePressed(MouseEvent e) {
        l.setText("Mouse Pressed at: " + e.getX() + ", " + e.getY());
    }

    public void mouseReleased(MouseEvent e) {
        l.setText("Mouse Released at: " + e.getX() + ", " + e.getY());
    }

    public void mouseEntered(MouseEvent e) {
        l.setText("Mouse Entered");
    }

    public void mouseExited(MouseEvent e) {
        l.setText("Mouse Exited");
    }   

    public static void main(String[] args) {
        new awt1();
    }
}