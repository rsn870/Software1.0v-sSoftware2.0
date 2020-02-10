The aim is to contrast the simple logic based programming paradigm with a Deep Neural Network on the same toy problem .

We shall choose multiple problems to contrast the approach (of course the problem(s) should be solvable by
both approaches ! )

The first problem is a multi-class classification problem the specifics of which are detailed below .

I term it as the "fizzbuzz" problem 

Given any number between 1-100 as input the output should be :
1) If the number is a multiple of 3 fizz should be printed followed by a new line


2) If the number is a multiple of 5 buzz should be printed followed by a new line


3) If the number is a multiple of 15 fizzbuzz should be printed followed by a new line


4) If the number does not belong to any of these categories it should be printed as it is followed by a
new line


Clearly mutliple if-else methods should do the trick ! But since we wish to see what deep learning can do
multiple neural networks are constructed (the simple feed forward architechture ) is used . Now , since such a simple 
architecture we are forced to some elaborate feature engineering (There is a reason why traditional methods haven't
died yet !)

Have fun messing around with the parameters and see how the accuracy changes !

Once you run the model script it will save the model in some h5 file which you can import to a main python file and then run as you please . A sample main.py which takes inputs of numbers from a text file (we are attaching a test_input.txt in case you folks need one ! ) and creating two new output files each for one pardigm has been attached as well 

The code is entirely in python the deep learning part has been done in tensorflow 2.0 

Happy Coding !
