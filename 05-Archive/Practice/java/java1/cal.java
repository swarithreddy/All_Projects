import java.awt.*;
import java.awt.event.*;

public class cal extends Frame implements ActionListener {

    TextField t1, t2, result;
    Button addButton, subButton, mulButton, divButton;

    cal() {
        // Initialize components
        t1 = new TextField();
        t2 = new TextField();
        result = new TextField();
        result.setEditable(false);

        addButton = new Button("Add");
        subButton = new Button("Subtract");
        mulButton = new Button("Multiply");
        divButton = new Button("Divide");

        // Set positions and size
        t1.setBounds(50, 50, 150, 20);
        t2.setBounds(50, 100, 150, 20);
        result.setBounds(50, 200, 150, 20);

        addButton.setBounds(50, 150, 70, 30);
        subButton.setBounds(130, 150, 70, 30);
        mulButton.setBounds(210, 150, 70, 30);
        divButton.setBounds(290, 150, 70, 30);

        // Add components to frame
        add(t1);
        add(t2);
        add(result);
        add(addButton);
        add(subButton);
        add(mulButton);
        add(divButton);

        // Add action listeners
        addButton.addActionListener(this);
        subButton.addActionListener(this);
        mulButton.addActionListener(this);
        divButton.addActionListener(this);

        // Frame settings
        setSize(400, 300);
        setLayout(null);
        setTitle("Simple Calculator");
        setVisible(true);

        // Handle window closing
        addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                dispose();
            }
        });
    }

    public void actionPerformed(ActionEvent e) {
        try {
            double num1 = Double.parseDouble(t1.getText());
            double num2 = Double.parseDouble(t2.getText());

            if (e.getSource() == addButton) {
                result.setText(String.valueOf(num1 + num2));
            } else if (e.getSource() == subButton) {
                result.setText(String.valueOf(num1 - num2));
            } else if (e.getSource() == mulButton) {
                result.setText(String.valueOf(num1 * num2));
            } else if (e.getSource() == divButton) {
                if (num2 != 0) {
                    result.setText(String.valueOf(num1 / num2));
                } else {
                    result.setText("Cannot divide by zero");
                }
            }
        } catch (NumberFormatException ex) {
            result.setText("Invalid input");
        }
    }

    public static void main(String[] args) {
        new cal();
    }
}
