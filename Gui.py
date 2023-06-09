import tkinter as tk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tkinter import *
from tkinter.ttk import *
import pyperclip 

root = tk.Tk()
root.configure(bg="#333333")
root.geometry("600x600")

tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")

entry = tk.Text(root, height=30, bg="#222222", fg="#ffffff")
entry.pack(side="top", fill="both", expand=True, padx=10, pady=10)
entry.pack_propagate(0)

def copy():
    print("copied")
    pyperclip.copy(label["text"])

def generate():
    input_text = entry.get("1.0", "end")
    listToStr = ' '.join([str(elem) for elem in input_text])
    count = sum(1 for char in listToStr if char.isalnum())
    
    # Fix the syntax error here by replacing the curly braces with a colon
    if count < 500:
        beams = 3
    
    if count > 500:
        beams = 1

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    # Use the value of `beams` that was determined based on `count`
    outputs = model.generate(input_ids, max_length=int(count / 6), min_length=int(count / 9), top_p=1, do_sample=True, num_beams=beams)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace("<n>", "\n")
    label.config(text=response)
    print(count)



def button_command():
    generate()

button = tk.Button(root, text="Summarize the Text", width=50, command=button_command, highlightthickness=2, highlightbackground = "blue", height=5, bg="#1c1c1c", fg="white")
button.pack(side="top", fill="both", expand=True, padx=10, pady=10)
button.pack_propagate(0)

label = tk.Label(root, text="Click The Button to Summarize The Text", borderwidth=2, relief="groove", highlightthickness=3, highlightbackground = "green", height=10, bg="#1c1c1c", fg="white")
label.pack(side="top", fill="both", expand=True, padx=10, pady=10)
label.pack_propagate(0)

button1 = tk.Button(root, text="Copy The Summarized Text", width=20, command=copy, highlightthickness=3, highlightbackground = "red", bg="#1c1c1c", fg="white")
button1.pack(side="top", expand=True, padx=10, pady=10)
button1.pack_propagate(0)

canvas = tk.Canvas(root, borderwidth=0, highlightthickness=0, bg="#333333")
canvas.pack(side="top", fill="both", expand=True)
canvas.pack_propagate(0)

root.mainloop()
