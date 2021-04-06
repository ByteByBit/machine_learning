import os.path

from tkinter import Tk, Canvas, Button, messagebox, LEFT, BOTH, TOP, Label
from tkinter import messagebox
from tkinter.ttk import Progressbar

from ml import DigitRecognizer
import threading


class Config:

    TITLE = "Draw 2 Predict"
    WIDTH = 600
    HEIGHT = 600
    DATA_FILE = 'model7.h5'


class MainUI(Canvas, DigitRecognizer):

    ''' Class to display and manage UI related tasks. '''

    def __init__(self):

        self.parent = Tk()
        self.c = Config()
        self.parent.title(self.c.TITLE)
        self.is_down = False
        self.x_position, self.y_position = None, None

        DigitRecognizer.__init__(self)


    def add_btns(self):
        ''' Adds Clear and Predict buttons to the UI.'''

        btn = Button(self.parent,fg="black",text="Predict",command=lambda: self.process())
        btn.pack(side=LEFT)
        btn = Button(self.parent,fg="black",text="Clear",command=lambda:self.drawing_area.delete('all'))
        btn.pack(side=LEFT)

    def add_canvas(self):
        ''' Adding canvas to the UI & canvas events. '''

        self.drawing_area = Canvas(self.parent,width=self.c.WIDTH,height=self.c.HEIGHT, bg="black")
        self.drawing_area.pack()
        # Events.
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.mouse_1_down)
        self.drawing_area.bind("<ButtonRelease-1>", self.mouse_1_up)

    def add_progressbar(self):
        ''' Adding progress bar to the UI. '''

        self.progress_bar = Progressbar(self.parent,orient ="horizontal",length = 500, mode='indeterminate')
        self.progress_bar.pack(expand=True, fill=BOTH, side=TOP)
        # Start loading...
        self.progress_bar.start()

    def create_ui(self):
        ''' Create UI. '''

        # Having the model already trained.
        if os.path.isfile(self.c.DATA_FILE):
            self.add_canvas()
            self.add_btns()

        # Show a progress bar till the training finished.
        else:
            self.label = Label(self.parent, text='Training...')
            self.label.pack()
            self.add_progressbar()
            self.loading()
            self.parent.after(100, self.start_training)
            self.parent.after(1000, self.process_q)


    def loading(self):
        ''' Loading function to remove progress bar, when training finished. '''

        if not self.progress:
            self.parent.after(1000, self.loading)

        else: # Training finished, destroy that bar...
            self.progress_bar.destroy()
            self.label.destroy()
            return self.create_ui()

    def display_msg(self, title: str, text: str):
        ''' Display message to the user. '''

        messagebox.showinfo(title, text)

    def get_drawing_area(self):
        ''' Get drawing area to save image. '''
        
        x = self.parent.winfo_rootx() + self.drawing_area.winfo_x()
        y = self.parent.winfo_rooty() + self.drawing_area.winfo_y()
        x1 = x + self.drawing_area.winfo_width()
        y1 = y + self.drawing_area.winfo_height()

        return x, x1, y, y1

    def mouse_1_down(self, event):
        ''' User pressed the left button. '''

        self.is_down = True

    def mouse_1_up(self, event):
        ''' User released left button. '''

        self.is_down = False
        self.x_position = None
        self.y_position = None

    def motion(self, event):
        ''' User keeps left button down -> drawing. '''

        if self.is_down:
            if self.x_position is not None and self.y_position is not None:

                # Display the line.
                event.widget.create_line(
                    self.x_position, self.y_position, event.x, event.y , width=25, fill="white")  

            # Update mouse position.
            self.x_position = event.x
            self.y_position = event.y