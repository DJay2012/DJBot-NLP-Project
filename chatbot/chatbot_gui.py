from tkinter import *
from tkinter import ttk
import json
#import the training.py
#and testing.py file
import testing as testpy
import training as trainpy

BG_GRAY="#ABB2B9"
BG_COLOR="#000"
TEXT_COLOR="#FFF"
FONT="Helvetica 14"
FONT_BOLD="Helvetica 13 bold"

class ChatBot:
    def __init__(self):
        #initialize tkinter window
        self.window=Tk()
        self.main_window()
        self.test=testpy.Testing()
        
    #run window
    def run(self):
        self.window.mainloop()
    
    def main_window(self):
        #add title to window and configure it
        self.window.title("ChatBot")
        self.window.resizable(width=False,height=False)
        self.window.configure(width=520,height=520,bg=BG_COLOR)
        #add tab for Chatbot and Train Bot in Notebook frame
        self.tab = ttk.Notebook(self.window)
        self.tab.pack(expand=1,fill='both')
        self.bot_frame=ttk.Frame(self.tab,width=520,height=520)
        self.train_frame=ttk.Frame(self.tab,width=520,height=520)
        self.tab.add(self.bot_frame,text='DJBot'.center(100))
        self.tab.add(self.train_frame,text='Train DJBot'.center(100))
        self.add_bot()
        self.add_train()
        
    def add_bot(self):
        #Add heading to the Chabot window
        head_label=Label(self.bot_frame,bg=BG_COLOR,fg=TEXT_COLOR,text="Welcome to DJay's Bot App",font=FONT_BOLD,pady=10)
        head_label.place(relwidth=1)
        line = Label(self.bot_frame,width=450,bg=BG_COLOR)
        line.place(relwidth=1,rely=0.07,relheight=0.012)

        #create text widget where conversation will be displayed
        self.text_widget=Text(self.bot_frame,width=20,height=2,bg="#fff",fg="#000",font=FONT,padx=5,pady=5)
        self.text_widget.place(relheight=0.745,relwidth=1,rely=0.08)
        self.text_widget.configure(cursor="arrow",state=DISABLED)

        #create scrollbar
        scrollbar=Scrollbar(self.text_widget)
        scrollbar.place(relheight=1,relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        #create bottom label where message widget will placed
        bottom_label=Label(self.bot_frame,bg=BG_GRAY,height=80)
        bottom_label.place(relwidth=1,rely=0.825)
        #this is for user to put query
        self.msg_entry=Entry(bottom_label,bg="#2C3E50",fg=TEXT_COLOR,font=FONT)
        self.msg_entry.place(relwidth=0.788,relheight=0.06,rely=0.008,relx=0.008)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>",self.on_enter)
        #send button which will call on_enter function to send the query
        send_button=Button(bottom_label,text="Send",font=FONT_BOLD,width=8,bg="Green",command=lambda: self.on_enter(None))   
        send_button.place(relx=0.80,rely=0.008,relheight=0.06,relwidth=0.20)

    def add_train(self):
        #Add heading to the Train Bot window
        head_label=Label(self.train_frame,bg=BG_COLOR,fg=TEXT_COLOR,text="Train DJBot",font=FONT_BOLD,pady=10)
        head_label.place(relwidth=1)

        #Tag Label and Entry for intents tag. 
        taglabel=Label(self.train_frame,fg="#000",text="Tag",font=FONT)
        taglabel.place(relwidth=0.2,rely=0.14,relx=0.008)
        self.tag=Entry(self.train_frame,bg="#fff",fg="#000",font=FONT)
        self.tag.place(relwidth=0.7,relheight=0.06,rely=0.14,relx=0.22)

        #Pattern Label and Entry for pattern to our tag.
        self.pattern=[]
        for i in range(2):
            patternlabel=Label(self.train_frame,fg="#000",text="Pattern%d"%(i+1),font=FONT)
            patternlabel.place(relwidth=0.2,rely=0.28+0.08*i,relx=0.008)
            self.pattern.append(Entry(self.train_frame,bg="#fff",fg="#000",font=FONT))
            self.pattern[i].place(relwidth=0.7,relheight=0.06,rely=0.28+0.08*i,relx=0.22)

        #Response Label and Entry for response to our pattern.
        self.response=[]
        for i in range(2):
            responselabel=Label(self.train_frame,fg="#000",text="Response%d"%(i+1),font=FONT)
            responselabel.place(relwidth=0.2,rely=0.50+0.08*i,relx=0.008)
            self.response.append(Entry(self.train_frame,bg="#fff",fg="#000",font=FONT))
            self.response[i].place(relwidth=0.7,relheight=0.06,rely=0.50+0.08*i,relx=0.22)

        #to train our bot create Train Bot button which will call on_train function
        bottom_label=Label(self.train_frame,bg=BG_GRAY,height=80)
        bottom_label.place(relwidth=1,rely=0.825)

        train_button=Button(bottom_label,text="Train Bot",font=FONT_BOLD,width=12,bg="Green",command=lambda: self.on_train(None))
        train_button.place(relx=0.20,rely=0.008,relheight=0.06,relwidth=0.60)
    
    def on_train(self,event):
        #read intent file and append created tag,pattern and responses from add_train function
        with open('intents.json','r+') as json_file:
            file_data=json.load(json_file)
            file_data['intents'].append({
            "tag": self.tag.get(),
            "patterns": [i.get() for i in self.pattern],
            "responses": [i.get() for i in self.response],
            "context": ""
            })
            json_file.seek(0)
            json.dump(file_data, json_file, indent = 1)
        #run and compile model from our training.py file.
        train=trainpy.Training()
        train.build(); print("Trained Successfully")
        self.test=testpy.Testing()
        
    def on_enter(self,event):
        #get user query and bot response
        msg=self.msg_entry.get()
        self.my_msg(msg,"You")
        self.bot_response(msg,"Bot")
        
    def bot_response(self,msg,sender):
        self.text_widget.configure(state=NORMAL)
        #get the response for the user's query from testing.py file
        self.text_widget.insert(END,str(sender)+" : "+self.test.response(msg)+"\n\n")
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)
    
    def my_msg(self,msg,sender):
        #it will display user query and bot response in text_widget
        if not msg:
            return
        self.msg_entry.delete(0,END)
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END,str(sender)+" : "+str(msg)+"\n")
        self.text_widget.configure(state=DISABLED)
        
# run the file
if __name__=="__main__":
    bot = ChatBot()
    bot.run()
