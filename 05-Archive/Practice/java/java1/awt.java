
import java.awt.*;
import java.awt.event.*;


public class awt extends Frame implements KeyListener{
    Label l;
    TextField t;
    awt(){
        l=new Label("enter text");
        l.setBounds(20,50,100,20);
        t=new TextField();
        t.setBounds(20,80,100,20);
        t.addKeyListener(this);
        add(l);
        add(t);
        setSize(300,300);   
        setLayout(null);
        setVisible(true);   

    }
    public void keyPressed(KeyEvent e) {
        l.setText("Key Pressed: " + e.getKeyChar());
    }
    public void keyReleased(KeyEvent e) {
        l.setText("Key Released: " + e.getKeyChar());
    }
    public void keyTyped(KeyEvent e) {
        l.setText("Key Typed: " + e.getKeyChar());
    }
    public static void main(String[] args) {
        new awt();
    }
}
