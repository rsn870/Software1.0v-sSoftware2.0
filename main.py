import tensorflow as tf
import numpy as np
import sys
import os


from tensorflow.keras.models import load_model






num_digits=16






trainer = load_model('model.h5')

#trainer.summary()


filename = sys.argv[-1]

inputa = np.loadtxt(filename , int , delimiter ='\n')

def inputvec(N):
  return np.array([N >> d & 1 for d in range(num_digits)])

def fizz_vec(N) :
    if N % 15 == 0:
        return np.array([1, 0, 0, 0])
    elif N % 5 == 0:
        return np.array([0, 1, 0, 0])
    elif N % 3 == 0:
        return np.array([0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 1])


l1 = [int(s) for s in inputa]
X_real = np.array([inputvec(i) for i in l1])
y_real = np.array([fizz_vec(i) for i in l1])


predictions = trainer.predict(X_real)
evaluations = trainer.evaluate(X_real,y_real)





repo_path = os.path.dirname(os.path.abspath(__file__))
out_sw_1_fp = repo_path + "/Software1.txt"
out_sw_2_fp = repo_path + "/Software2.txt"



f = open(out_sw_1_fp,'a')
for i in l1 :
     
      if i % 15 == 0: 
       f.write('fizzbuzz' + '\n')
      elif i % 5 == 0: 
       f.write('buzz' + '\n')
      elif i % 3 == 0: 
       f.write('fizz' + '\n')
      else : 
       f.write(str(i) + '\n')








g = open(out_sw_2_fp,'a')
for i in range (1,len(l1)+1) :
  j = np.unravel_index(predictions[i-1].argmax(), predictions[i-1].shape)
  
  if j == (0,) :
    g.write("fizzbuzz " +  "\n")
  elif j == (1,) :
    g.write("buzz "+ "\n")
  elif j == (2,) :
     g.write("fizz" + "\n")
  else :
     g.write(str(i) + "\n")


  

     


