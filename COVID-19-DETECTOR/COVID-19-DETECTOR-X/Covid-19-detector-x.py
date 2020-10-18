import numpy as np
from numpy import sqrt
from numpy import argmax
import cv2
import tensorflow as tf
import glob
import time
from PIL import Image as im
from PIL import ImageTk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from sklearn.metrics import confusion_matrix


def click1(path):
  start=time.time()
  interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  image =cv2.imread(path)
  input_data= np.array(np.expand_dims(prepare(image),0), dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data[0])
  interpreter.invoke()
  printimage(image)
  output_data = interpreter.get_tensor(output_details[0]['index'])
  res=np.argmax(output_data,axis=1)[0]
  accuracy=float(np.max(output_data,axis=1)[0])
  if res == 0 :
    result.config(text ="Negative\t Probability : "+str(int(accuracy*100))+"%")
  else : 
    result.config(text ="Positive\t Probability : "+str(int(accuracy*100))+"%")
  end=time.time()
  print("time elapsed : "+str(end-start))
'''
  if output_data[0][0].astype(int)==0 :
    result.config(text='Negative')
    print(output_data)
  else:
    result.config(text='Positive')
    print(output_data)
  print(res)
'''

def plotImages(images_arr):
    fig, axes = plt.subplots(1,1, figsize=(5,5))
    #axes = axes.flatten()
    for img, ax in zip(images_arr, [axes]):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def prepare(ima):
  IMG_SIZE = 100
  img_array = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
  img_array=img_array/255.0
  new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  return new_array.reshape(-1,IMG_SIZE, IMG_SIZE,1)

def printimage(ima):
  img_array = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
  

def getPath():
  p=filedialog.askopenfilename(filetypes=(("Image File",'*.jpg'),("Image File",'*.jpeg'),("Image File",'*.png'),("Image File",'*.tif'),("Image File",'*.bmp')))
  entry.delete(0,END)
  entry.insert(0,p)
  img=im.open(p)
  img=img.resize((350,350),im.ANTIALIAS)
  photo=ImageTk.PhotoImage(img)
  label.config(image = photo)
  label.image=photo
  result.config(text="........")

def getPath2():
  p=filedialog.askdirectory()
  entry2.delete(0,END)
  entry2.insert(0,p)
  listbox.delete(0,END)
  for i in glob.glob(p+"/*"):
    k=i.split('/')
    k=k[-1].split('.')
    listbox.insert(END,k[0]+ " : ")

def click2(path):
  start=time.time()
  n=0
  p=0
  counter=0
  accuracy=0.0
  #actual=[]
  #predict=[]
  listbox.delete(0,END)
  for i in glob.glob(path+"/*"):
    interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image =cv2.imread(i)
    input_data= np.array(np.expand_dims(prepare(image),0), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data[0])
    interpreter.invoke()
    printimage(image)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    res=np.argmax(output_data,axis=1)[0]
    k=i.split('/')
    k=k[-1].split('.')
    t=(k[0].split())[0]
    accuracy+=float(np.max(output_data,axis=1)[0])
    #if t == 'Negative' :
     #actual.append(0)
    #else :
      #actual.append(1)
    if res == 0 :
      listbox.insert(counter,k[0]+ " : Negative\t Prob. : "+str(int(float(np.max(output_data,axis=1)[0])*100))+"%")
      counter+=1
      n+=1
      #predict.append(0)
    else :
      listbox.insert(counter,k[0]+ " : Positive\t Prob. : "+str(int(float(np.max(output_data,axis=1)[0])*100))+"%")
      counter+=1
      p+=1
      #predict.append(1)
  active_label.config(text='Positive : '+str(p)+'\tNegative : '+str(n)+"\t Average Probability : "+str(int((accuracy/(counter+1))*100))+"%")
  end=time.time()
  print("time elapsed : "+str(end-start))
'''
  tn, fp, fn, tp = confusion_matrix(actual, predict).ravel()
  acc = float((tp+tn)/len(actual))
  mis_rate = float((fp+fn)/len(actual))
  true_positive_rate = float(tp/actual.count(1))
  false_positive_rate = (float(fp/actual.count(0)))
  true_negative_rate = (float(tn/actual.count(0)))
  precision = (float(tp/predict.count(1)))
  prevalence = (float(actual.count(1)/len(actual)))
  detection_rate = (float(tp/(tp+fp)))
  recall = (float(tp/(tp+fn)))
  fscore=(2*precision*recall)/(precision+recall)
  print("Accuracy : "+str(acc)+"\nMisclassification Rate : "+str(mis_rate)+"\nTrue Positive Rate : "+str(true_positive_rate)+"\nFalse Positive Rate : "+str(false_positive_rate)+"\nTrue Negative Rate : "+str(true_negative_rate)+"\nPrecision : "+str(precision)+"\nPrevalence : "+str(prevalence)+"\nDetection Rate : "+str(detection_rate))
'''


root = Tk()
root.attributes("-fullscreen",True)
root.configure(bg="#0066ff")
style1 = ttk.Style()
style1.configure('TFrame', background = 'white',height = 500,width=400)
style1.configure('TNotebook',background = 'white')
style1.configure('TNotebook.Tab',background="white",foreground="#0066ff",font = ('Boulder', 12, 'bold'),lightcolor='white',border=0,selectedbackground="#0066ff",selectedforeground="white")
notebook = ttk.Notebook(root)
notebook.pack(side=TOP,pady = 10)
Label(root,text = "COVID-19 XRAY DETECTOR X",font=("times",26,"bold"),bg="#0066ff",fg="white").pack(side=TOP,pady=60)
frame1 = ttk.Frame(notebook,relief = RIDGE)
frame1.pack(fill=BOTH)
frame2 = ttk.Frame(notebook,relief = RIDGE)
frame2.pack(fill=BOTH,padx = 10, pady = 10)
frame3 = ttk.Frame(notebook,relief = RIDGE)
frame3.pack(fill=BOTH,padx = 10, pady = 10)
notebook.add(frame1, text = 'Import One Image')
notebook.add(frame2, text = 'Import Multi Image')
notebook.add(frame3, text = 'Scanner')
entry = Entry(frame1,background = 'white',font=("Boulder",10,"bold"),width=30)
entry.pack(side=LEFT,padx = 10, pady = 10)
choose=Button(frame1, text = 'Import',command= getPath,width=4,height=1,bg="#0066ff",fg="white",font=("Boulder",10,"bold"),activebackground="white",activeforeground="#0066ff",border = 0)
choose.pack(side=LEFT,padx = 5, pady = 10)
test=Button(frame1, text = 'Test Image',command= lambda : click1(entry.get()),width=10,height=5,bg="#0066ff",fg="white",font=("Boulder",12,"bold"),activebackground="white",activeforeground="#0066ff",border = 0)
test.pack(side=LEFT,padx = 30, pady = 10)
label=Label(frame1,text="No Image To Show",font=("Boulder",14,"bold"),bg="white",fg="#0066ff")
label.pack(side=BOTTOM,padx=30,pady=30)
result=Label(frame1,text='No Result To Show',font=("Boulder",14,"bold"),bg="white",fg="#0066ff")
result.pack(side=BOTTOM,padx=30,pady=30)
entry2 = Entry(frame2,background = 'white',font=("Boulder",10,"bold"),width=30)
entry2.pack(side=LEFT,padx = 10, pady = 10)
choose2=Button(frame2, text = 'Import',command= getPath2,width=4,height=1,bg="#0066ff",fg="white",font=("Boulder",10,"bold"),activebackground="white",activeforeground="#0066ff",border = 0)
choose2.pack(side=LEFT,padx = 5, pady = 10)
test2=Button(frame2, text = 'Test All',command= lambda : click2(entry2.get()),width=10,height=5,bg="#0066ff",fg="white",font=("Boulder",12,"bold"),activebackground="white",activeforeground="#0066ff",border = 0)
test2.pack(side=LEFT,padx = 30, pady = 10)
result_frame=LabelFrame(frame2,text = 'Results',bg="#0066ff",fg="white",font=("times",16,"bold"))
scrollbar = Scrollbar(result_frame,orient=VERTICAL,background="white",activebackground="white",troughcolor="#0066ff")
listbox=Listbox(result_frame,yscrollcommand=scrollbar.set,bg="#0066ff",fg="white",font=("times",14,"bold"),width=35,height=15,selectbackground="white",selectforeground="#0066ff")
scrollbar.config(command=listbox.yview)
scrollbar.pack(side=RIGHT,fill=Y)
result_frame.pack(side=BOTTOM,padx=30,pady=30)
listbox.pack()
active_label=Label(frame2,text='........',font=("Boulder",14,"bold"),bg="white",fg="#0066ff")
active_label.pack(side=BOTTOM,padx=30,pady=30)
exit=Button(root, text = 'Exit',command=root.quit,width=20,height=2,bg="white",fg="#0066ff",font=("Boulder",14,"bold"),activebackground="#0066ff",activeforeground="white",border = 0)
exit.pack(side = BOTTOM,padx = 10, pady = 10)
root.mainloop()
