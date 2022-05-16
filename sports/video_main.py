

from tkinter import filedialog
from tkinter import Tk,Toplevel,PhotoImage,BOTH,RAISED,Canvas,CENTER,Canvas,Label,Frame,Checkbutton,Button,LEFT,BOTTOM,X,Y,E,W,Entry,Menu,TOP,SUNKEN
from tkinter import *
import os
def donothing():
    print('nothing')
    
    

root=Tk()
root.geometry("640x640")


def run_video():
    os.system('python run_video.py')


def run_video_tiny():
    os.system('python run_video_tiny.py')

def run_camera():
    os.system('python run_camera.py')

def run_camera_tiny():
    # sub=Toplevel(root)
    # sub.geometry("480x480")
    os.system('python run_camera_tiny.py')


toolbar=Frame(root,bg='indigo',bd=2)
insertbutt=Button(toolbar,text='VIDEO FAST',command=run_video_tiny)
insertbutt.pack(side=LEFT,padx=2,pady=2,ipadx=20)
printexit1=Button(toolbar,text='CAMERA FAST',command=run_camera_tiny)
printexit1.pack(side=LEFT,padx=2,pady=2,ipadx=20)

insertbutt=Button(toolbar,text='VIDEO ACCURATE',command=run_video)
insertbutt.pack(side=LEFT,padx=2,pady=2,ipadx=20)
printexit1=Button(toolbar,text='CAMERA ACCURATE',command=run_camera)
printexit1.pack(side=LEFT,padx=2,pady=2,ipadx=20)



toolbar.pack(side=TOP,fill=BOTH)

# toolbar1=Frame(root,bg='red')
# toolbar1=Label(root,bg='indigo',bd=2)
# insertbutt=Button(toolbar,text='run_video_tiny',command=run_video_tiny)
# insertbutt.pack(side=LEFT,padx=2,pady=2,ipadx=20)
# printexit1=Button(toolbar,text='run_camera_tiny',command=run_camera_tiny)
# printexit1.pack(side=LEFT,padx=2,pady=2,ipadx=20)

# insertbutt1=Button(toolbar1,text='run_video',command=run_video)
# insertbutt1.pack(side=LEFT,padx=2,pady=2,ipadx=20)
# printexit11=Button(toolbar1,text='run_camera',command=run_camera)
# printexit11.pack(side=LEFT,padx=2,pady=2,ipadx=20)


#canvas=Canvas(root,width=300,height=20)

explanation = """REAL TIME  DETECTION"""


w = Label(root, 
             compound = CENTER,
             text=explanation,bg='light green',pady=3,
             font=('Courier',20,'bold')
             ).pack(fill=BOTH,expand=True)



     

logo = PhotoImage(file="person.png")

explanation = """ DETECTIONS FOR PERSONS"""


canvas=Canvas(root,width=300,height=20)


w = Label(root, 
             compound = BOTTOM,
             text=explanation,font=('Courier',20,'bold'),bg='white',
             
             image=logo).pack(fill=BOTH,expand=True)

# canvas.pack(expand=True)




status=Label(root,text=' Video Processing .... ',bd=3,relief=SUNKEN,anchor=W)
status.pack(side=BOTTOM,fill=BOTH)

# toolbar1.pack(side=BOTTOM,fill=BOTH)




def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


button1 = Button(root,bg='light blue',bd=3, text="Quit", command=_quit)
button1.pack(side=BOTTOM,fill=BOTH)


root.mainloop()

