using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MNIST_Supervised_Learning
{
    public partial class Form1 : Form
    {
        bool mouseDown = false;
        Network neuralNet;
        List<Button> buttons = new List<Button>();
        public Form1(Network mainNN)
        {
            InitializeComponent();
            int buttonEdgeLength = 10;
            int startX = 0;
            int startY = 0;

            for (int rowIdx = 0; rowIdx < 28; rowIdx++)
            {
                for (int colIdx = 0; colIdx < 28; colIdx++)
                {
                    Button newButton = new Button
                    {
                        Location = new System.Drawing.Point(startX + rowIdx * buttonEdgeLength, startY + colIdx * buttonEdgeLength),
                        Name = "button" + (rowIdx + colIdx),
                        Size = new System.Drawing.Size(buttonEdgeLength, buttonEdgeLength),
                        TabIndex = 0,
                        BackColor = Color.White,
                        UseVisualStyleBackColor = true
                    };
                    //newButton.MouseEnter += new System.EventHandler(button_Enter);
                    newButton.MouseUp += new System.Windows.Forms.MouseEventHandler(button_Down);
                    buttons.Add(newButton);
                    Controls.Add(newButton);
                }
            }

            //MouseDown += onMouseDown;
            //MouseUp += onMouseUp;

            neuralNet = mainNN;
        }

        private void button_Enter(object sender, EventArgs e)
        {
            if (mouseDown)
            {
                Button button = (Button)sender;
                button.BackColor = System.Drawing.Color.Black;
            }
        }


        private void button_Down(object sender, EventArgs e)
        {
            Button button = (Button)sender;
            if (button.BackColor == System.Drawing.Color.White)
                button.BackColor = System.Drawing.Color.Black;
            else
                button.BackColor = System.Drawing.Color.White;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            double[] inputs = new double[28 * 28];
            int idx = 0;
            buttons.ForEach(button =>
            {
                if (button.BackColor == System.Drawing.Color.White)
                    inputs[idx] = 0;
                else
                    inputs[idx] = 1;

                idx++;
            });

            double[] output = neuralNet.forwardPropagate(inputs);
            this.label2.Text = "" + output.ToList().IndexOf(output.Max());
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void label2_Click(object sender, EventArgs e)
        {

        }

        //private void onMouseDown(object sender, MouseEventArgs e)
        //{
        //    Control control = (Control)sender;

        //    if (control.Capture)
        //        control.Capture = false;

        //    if (e.Button == MouseButtons.Left)
        //    {
        //        mouseDown = true;
        //        Console.WriteLine(mouseDown);
        //    }

        //}

        //private void onMouseUp(object sender, MouseEventArgs e)
        //{
        //    Control control = (Control)sender;

        //    if (!control.Capture)
        //        control.Capture = true;

        //    if (e.Button == MouseButtons.Left)
        //    {
        //        mouseDown = false;
        //        Console.WriteLine(mouseDown);
        //    }
        //}
    }
}
