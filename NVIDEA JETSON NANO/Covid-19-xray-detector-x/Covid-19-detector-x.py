import sys
import os
import numpy as np
from numpy import sqrt
from numpy import argmax
import cv2
import tensorflow as tf
import tensorflow.math
import glob
import time
from PIL import Image as im
from PIL import ImageTk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from sklearn.metrics import confusion_matrix


def restart_program():
    python = sys.executable
    os.execl(python, python, * sys.argv)


def ExitApplication():
    MsgBox = messagebox.askquestion('Exit Application','Are you sure you want to exit the application',icon = 'warning')
    if MsgBox == 'yes':
       root.destroy()

def load():
  try:
    file=open("/home/nvidea/Covid-19-xray-detector-x/config","r")
    line=file.readlines()[0].split()
    file.close()
    return line
  except IOError:
    messagebox.showerror("Error", "config file corrupted or doesn't exist !!")
    return None

def save(fullscreen,theme,covid19,pneumonia,gender,time):
  try:
    file=open("/home/nvidea/Covid-19-xray-detector-x/config","w")
    file.truncate(0)
    file.write(str(fullscreen)+" "+str(theme)+" "+str(covid19)+" "+str(pneumonia)+" "+str(gender)+" "+str(time))
    file.close()
    messagebox.showinfo("Info", "Settings Saved, Please restart the application to take the all changes!")
    MsgBox = messagebox.askquestion('Restart Application','Want to restart application now ?',icon = 'warning')
    if MsgBox == 'yes':
      restart_program()
  except IOError:
    messagebox.showerror("Error", "config file corrupted or doesn't exist !!")

def saveData(fullscreen,theme,covid19,pneumonia,gender,time):
  save(fullscreen,theme,covid19,pneumonia,gender,time)
  
  
def click1(path):
  try:
    listbox2.delete(0,END)
    start=time.time()
    settings=load()
    image =cv2.imread(path)
    printimage(image)
    if settings[2] == '1':
      interpreter1 = tf.lite.Interpreter(model_path="/home/nvidea/Covid-19-xray-detector-x/models/covid19.tflite")
      interpreter1.allocate_tensors()
      input_details = interpreter1.get_input_details()
      output_details = interpreter1.get_output_details()
      input_data= np.array(np.expand_dims(prepare(image),0), dtype=np.float32)
      interpreter1.set_tensor(input_details[0]['index'], input_data[0])
      interpreter1.invoke()
      output_data = interpreter1.get_tensor(output_details[0]['index'])
      res1=np.argmax(output_data,axis=1)[0]
      prob1=float(np.max(output_data,axis=1)[0])
      if res1 == 0 :
        listbox2.insert(2," ")
        listbox2.insert(3," COVID-19  ||\t\tPositive: "+str(100-int(prob1*100))+"%\t\tNegative : "+str(int(prob1*100))+"%")
      else :
        listbox2.insert(2," ")
        listbox2.insert(3," COVID-19  ||\t\tPositive: "+str(int(prob1*100))+"%\t\tNegative : "+str(100-int(prob1*100))+"%")
    if settings[3] == '1':
      interpreter2 = tf.lite.Interpreter(model_path="/home/nvidea/Covid-19-xray-detector-x/models/pneumonia.tflite")
      interpreter2.allocate_tensors()
      input_details = interpreter2.get_input_details()
      output_details = interpreter2.get_output_details()
      input_data= np.array(np.expand_dims(prepare(image),0), dtype=np.float32)
      interpreter2.set_tensor(input_details[0]['index'], input_data[0])
      interpreter2.invoke()
      output_data = interpreter2.get_tensor(output_details[0]['index'])
      res2=np.argmax(output_data,axis=1)[0]
      prob2=float(np.max(output_data,axis=1)[0])
      if res2 == 0 :
        listbox2.insert(4," ")
        listbox2.insert(5," Pneumonia ||\t\tPositive: "+str(100-int(prob2*100))+"%\t\tNegative : "+str(int(prob2*100))+"%")
      else :
        listbox2.insert(4," ")
        listbox2.insert(5," Pneumonia ||\t\tPositive: "+str(int(prob2*100))+"%\t\tNegative : "+str(100-int(prob2*100))+"%")
    if settings[4] == '1':
      interpreter3 = tf.lite.Interpreter(model_path="/home/nvidea/Covid-19-xray-detector-x/models/gender.tflite")
      interpreter3.allocate_tensors()
      input_details = interpreter3.get_input_details()
      output_details = interpreter3.get_output_details()
      input_data= np.array(np.expand_dims(prepare(image),0), dtype=np.float32)
      interpreter3.set_tensor(input_details[0]['index'], input_data[0])
      interpreter3.invoke()
      output_data = interpreter3.get_tensor(output_details[0]['index'])
      res3=np.argmax(output_data,axis=1)[0]
      if res3 == 0 :
        listbox2.insert(0," ")
        listbox2.insert(1," Gender : Female")
      else :
        listbox2.insert(0," ")
        listbox2.insert(1," Gender : Male")
    end=time.time()
    if settings[5] == '1':
      et=round(end-start,2)
      listbox2.insert(8," ")
      listbox2.insert(9," Time Elapsed : "+str(et)+" s")
  except:
    messagebox.showwarning("Attention", "Please import a true path first!")


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
  img=img.resize((405,405),im.ANTIALIAS)
  photo=ImageTk.PhotoImage(img)
  label.config(image = photo)
  label.image=photo
  listbox2.delete(0,END)


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
  try :
    start=time.time()
    string=""
    counter=0
    #accuracy=0.0
    settings=load()
    #actual=[]
    #predict=[]
    listbox.delete(0,END)
    for i in glob.glob(path+"/*"):
      k=i.split('/')
      k=k[-1].split('.')
      t=(k[0].split())[0]
      if settings[4] == '1':
        interpreter = tf.lite.Interpreter(model_path="/home/nvidea/Covid-19-xray-detector-x/models/gender.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        image =cv2.imread(i)
        input_data= np.array(np.expand_dims(prepare(image),0), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data[0])
        interpreter.invoke()
        printimage(image)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        res1=np.argmax(output_data,axis=1)[0]
        prob1=float(np.max(output_data,axis=1)[0])
        if res1 == 1:
          string+="  Gender : Male ||  "
        else:
          string+="  Gender : Female ||  "	  
      if settings[2] == '1':
        interpreter = tf.lite.Interpreter(model_path="/home/nvidea/Covid-19-xray-detector-x/models/covid19.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        image =cv2.imread(i)
        input_data= np.array(np.expand_dims(prepare(image),0), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data[0])
        interpreter.invoke()
        printimage(image)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        res2=np.argmax(output_data,axis=1)[0]
        prob2=float(np.max(output_data,axis=1)[0])
        if res2 == 1:
           string+="COVID-19: Positive: "+str(int(prob2*100))+"%  Negative: "+str(100-int(prob2*100))+"% ||  "
        else:
          string+="COVID-19: Positive: "+str(100-int(prob2*100))+"%  Negative: "+str(int(prob2*100))+"% ||  "
      if settings[3] == '1':
        interpreter = tf.lite.Interpreter(model_path="/home/nvidea/Covid-19-xray-detector-x/models/pneumonia.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        image =cv2.imread(i)
        input_data= np.array(np.expand_dims(prepare(image),0), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data[0])
        interpreter.invoke()
        printimage(image)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        res3=np.argmax(output_data,axis=1)[0]
        prob3=float(np.max(output_data,axis=1)[0])
        if res3 == 1:
          string+="Pneumonia: Positive: "+str(int(prob3*100))+"%  Negative: "+str(100-int(prob3*100))+"% ||  "
        else:
          string+="Pneumonia: Positive: "+str(100-int(prob3*100))+"%  Negative: "+str(int(prob3*100))+"% ||  "
      listbox.insert(counter,k[0]+":||"+string)
      counter+=1
      #accuracy+=float(np.max(output_data,axis=1)[0])
      #if t == 'Negative' :
       #actual.append(0)
      #else :
        #actual.append(1)
      #if res == 0 :
        #listbox.insert(counter,k[0]+ " : Negative\t Prob. : "+str(int(float(np.max(output_data,axis=1)[0])*100))+"%")
        #counter+=1
        #predict.append(0)
      #else :
        #listbox.insert(counter,k[0]+ " : COVID-19\t Prob. : "+str(int(float(np.max(output_data,axis=1)[0])*100))+"%")
        #counter+=1
        #predict.append(1)
    end=time.time()   
    print("time elapsed : "+str(end-start))
  except :
    messagebox.showwarning("Attention", "Please import a true directory first!")
    #tn, fp, fn, tp = confusion_matrix(actual, predict).ravel()
    #acc = float((tp+tn)/len(actual))
    #mis_rate = float((fp+fn)/len(actual))
    #true_positive_rate = float(tp/actual.count(1))
    #false_positive_rate = (float(fp/actual.count(0)))
    #true_negative_rate = (float(tn/actual.count(0)))
    #precision = (float(tp/predict.count(1)))
    #prevalence = (float(actual.count(1)/len(actual)))
    #detection_rate = (float(tp/(tp+fp)))
    #recall = (float(tp/(tp+fn)))
    #fscore=(2*precision*recall)/(precision+recall)
    #print("Accuracy : "+str(acc)+"\nMisclassification Rate : "+str(mis_rate)+"\nTrue Positive Rate : "+str(true_positive_rate)+"\nFalse Positive Rate : "+str(false_positive_rate)+"\nTrue Negative Rate : "+str(true_negative_rate)+"\nPrecision : "+str(precision)+"\nPrevalence : "+str(prevalence)+"\nDetection Rate : "+str(detection_rate))

root = Tk()
settings=load()
v = IntVar()
t=IntVar()
cov=IntVar()
pneu=IntVar()
gen=IntVar()
ti=IntVar()
dark_img=im.open("/home/nvidea/Covid-19-xray-detector-x/themes/dark.png")
dark_img=dark_img.resize((300,200),im.ANTIALIAS)
dark_photo=ImageTk.PhotoImage(dark_img)
blue_img=im.open("/home/nvidea/Covid-19-xray-detector-x/themes/blue.png")
blue_img=blue_img.resize((300,200),im.ANTIALIAS)
blue_photo=ImageTk.PhotoImage(blue_img)
if settings[0] == '1':
  root.attributes("-fullscreen",True)
  v.set(1)
else:
  root.attributes("-fullscreen",False)
  v.set(0)
if settings[1] == '0':
  color1="#444444"
  color2="#878683"
  t.set(0)
elif settings[1] == '1':
  color1="#4e69aa"
  color2="#2b3c63"
  t.set(1)
if settings[2]=='0':
  cov.set(0)
else:
  cov.set(1)
if settings[3]=='0':
  pneu.set(0)
else:
  pneu.set(1)
if settings[4]=='0':
  gen.set(0)
else:
  gen.set(1)
if settings[5]=='0':
  ti.set(0)
else:
  ti.set(1)
root.configure(bg=color2)
style1 = ttk.Style()
style1.configure('TFrame', background = color1,weight=1)
style1.configure('TNotebook',background = color1)
style1.configure('TNotebook.Tab',background=color2,foreground="white",font = ('Boulder', 17, 'bold'),lightcolor=color1,border=0)
style1.map('TNotebook.Tab',foreground=[('selected', color1)],background=[('selected', 'white')])
notebook = ttk.Notebook(root)
notebook.pack(side=TOP,pady = 10)
Label(root,text = "COVID-19 XRAY DETECTOR X",font=("times",36,"bold"),bg=color2,fg="white").pack(side=TOP,pady=60)
frame1 = ttk.Frame(notebook,relief = RIDGE)
frame1.pack(side=TOP)
frame2 = ttk.Frame(notebook,relief = RIDGE)
frame2.pack(fill=BOTH,padx = 10, pady = 10)
frame3 = ttk.Frame(notebook,relief = RIDGE)
frame3.pack(fill=BOTH,padx = 10, pady = 10)
frame4 = ttk.Frame(frame1)
frame4.pack(side=TOP, pady = 30)
frame5 = ttk.Frame(notebook,relief = RIDGE)
frame5.pack(fill=BOTH,padx = 10, pady = 10)
frame6 =LabelFrame(frame5,text="Display",bg=color2,fg="white",font=("times",16,"bold"))
frame6.pack(side=TOP,fill=X,padx = 10, pady = 10)
frame7 =LabelFrame(frame5,text="Theme",bg=color2,fg="white",font=("times",16,"bold"))
frame7.pack(side=TOP,fill=X,padx = 10, pady = 10)
frame8 =LabelFrame(frame5,text="Results",bg=color2,fg="white",font=("times",16,"bold"))
frame8.pack(side=TOP,fill=X,padx = 10, pady = 10)
frame9 = ttk.Frame(frame2)
frame9.pack(side=TOP, pady = 30)
fullscreen_label=Label(frame6,text="Fullscreen Mode : ",font=("times",14,"bold"),bg=color2,fg="white")
fullscreen_label.pack(side=LEFT,padx=10,pady=10)
rb1=Radiobutton(frame6, text="ON", variable=v, value=1,font=("times",14,"bold"),bg=color2,fg=color1,indicator = 0,activebackground="white",activeforeground=color1,width=10,borderwidth=0,highlightcolor=color1).pack(side=LEFT,padx=20)
rb2=Radiobutton(frame6, text="OFF", variable=v, value=0,font=("times",14,"bold"),bg=color2,fg=color1,indicator = 0,activebackground="white",activeforeground=color1,width=10,borderwidth=0,highlightcolor=color1).pack(side=LEFT,padx=20)
rb3=Radiobutton(frame7,image=dark_photo, variable=t, value=0,borderwidth=0,highlightcolor=color1).pack(side=LEFT,padx=20)
rb4=Radiobutton(frame7, image=blue_photo, variable=t, value=1,borderwidth=0,highlightcolor=color1).pack(side=LEFT,padx=20)
checkb1=Checkbutton(frame8,text="COVID-19",font=("times",14,"bold"),bg=color2,fg=color1,variable = cov,onvalue = 1,offvalue = 0,borderwidth=0,highlightcolor=color1,indicator = 0,activebackground=color2,activeforeground=color1,width=20).pack(side=LEFT,padx=20)
checkb2=Checkbutton(frame8,text="Pneumonia",font=("times",14,"bold"),bg=color2,fg=color1,variable = pneu,onvalue = 1,offvalue = 0,borderwidth=0,highlightcolor=color1,indicator = 0,activebackground=color2,activeforeground=color1,width=20).pack(side=LEFT,padx=20)
checkb3=Checkbutton(frame8,text="Gender",font=("times",14,"bold"),bg=color2,fg=color1,variable = gen,onvalue = 1,offvalue = 0,borderwidth=0,highlightcolor=color1,indicator = 0,activebackground=color2,activeforeground=color1,width=20).pack(side=LEFT,padx=20)
checkb4=Checkbutton(frame8,text="Time Elapsed",font=("times",14,"bold"),bg=color2,fg=color1,variable = ti,onvalue = 1,offvalue = 0,borderwidth=0,highlightcolor=color1,indicator = 0,activebackground=color2,activeforeground=color1,width=20).pack(side=LEFT,padx=20)
Label(frame8,text="\n\n\n\n",bg=color2).pack(side=BOTTOM)
Button(frame5,text="Save",command= lambda : saveData(v.get(),t.get(),cov.get(),pneu.get(),gen.get(),ti.get()),width=20,height=2,bg="white",fg=color1,font=("times",16,"bold"),activebackground=color1,activeforeground="white",border = 0).pack(side=BOTTOM,pady=30)
notebook.add(frame1, text = 'Import One Image')
notebook.add(frame2, text = 'Import Multi Image')
notebook.add(frame3, text = 'Scanner')
notebook.add(frame5, text = 'Settings')
entry = Entry(frame4,background = color1,foreground="white",font=("Boulder",15,"bold"),width=50)
entry.pack(side=LEFT,pady=0)
choose=Button(frame4, text = 'Import',command= getPath,width=8,height=1,bg="white",fg=color1,font=("Boulder",15,"bold"),activebackground=color1,activeforeground="white",border = 0)
choose.pack(side=LEFT,padx = 10, pady = 0)
test=Button(frame1, text = 'Test Image',command= lambda : click1(entry.get()),width=15,height=5,bg="white",fg=color1,font=("Boulder",15,"bold"),activebackground=color1,activeforeground="white",border = 0)
test.pack(side=TOP,padx = 30, pady = 10)
label=Label(frame1,text="No Image To Show",font=("Boulder",15,"bold"),bg=color1,fg="white")
label.pack(side=LEFT,padx=40,pady=30)
result_frame2=LabelFrame(frame1,text = 'Results',bg=color2,fg="white",font=("times",16,"bold"))
scrollbar2 = Scrollbar(result_frame2,orient=VERTICAL,background=color1,activebackground=color1,troughcolor="white")
listbox2=Listbox(result_frame2,yscrollcommand=scrollbar2.set,bg=color2,fg="white",font=("times",16,"bold"),width=50,height=15,selectbackground=color1,selectforeground="white")
scrollbar2.config(command=listbox2.yview)
scrollbar2.pack(side=RIGHT,fill=Y)
result_frame2.pack(side=RIGHT,padx=40,pady=20)
listbox2.pack()
entry2 = Entry(frame9,background = color1,foreground="white",font=("Boulder",15,"bold"),width=50)
entry2.pack(side=LEFT,padx = 10, pady = 10)
choose2=Button(frame9, text = 'Import',command= getPath2,width=8,height=1,bg="white",fg=color1,font=("Boulder",15,"bold"),activebackground=color1,activeforeground="white",border = 0)
choose2.pack(side=LEFT,padx = 5, pady = 10)
test2=Button(frame2, text = 'Test All',command= lambda : click2(entry2.get()),width=10,height=5,bg="white",fg=color1,font=("Boulder",12,"bold"),activebackground=color1,activeforeground="white",border = 0)
test2.pack(side=TOP, pady = 20)
result_frame=LabelFrame(frame2,text = 'Results',bg=color2,fg="white",font=("times",16,"bold"))
scrollbar = Scrollbar(result_frame,orient=VERTICAL,background=color1,activebackground=color1,troughcolor="white")
listbox=Listbox(result_frame,yscrollcommand=scrollbar.set,bg=color2,fg="white",font=("times",14,"bold"),width=100,height=15,selectbackground=color1,selectforeground="white")
scrollbar.config(command=listbox.yview)
scrollbar.pack(side=RIGHT,fill=Y)
result_frame.pack(side=BOTTOM,padx=10,pady=20)
listbox.pack()
coming_soon=Label(frame3,text=".... Coming Soon ....",font=("Boulder",18,"bold"),bg=color1,fg="white")
coming_soon.pack(side=TOP,padx=50,pady=50)
exit=Button(root, text = 'Exit',command=ExitApplication,width=20,height=2,bg="white",fg=color1,font=("Boulder",14,"bold"),activebackground=color2,activeforeground="white",border = 0)
exit.pack(side = BOTTOM,padx = 10, pady = 10)
root.mainloop()
