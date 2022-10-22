from tkinter import *
window=Tk()
window.geometry("700x700")
photo=PhotoImage(file="cctv2.png")
no_root=Label(image=photo)
no_root.pack()
f1=Frame(window,borderwidth=3,bg="white")
f1.pack(side="left")
b1=Button(f1,fg="red",text="Click to only record video")
b1.pack(side="bottom", pady=30)
b2=Button(f1,fg="red",text="Click to see video with known and unkown users")
b2.pack(side="bottom")
b3=Button(f1,fg="red",text="Click to see video along with motion")
b3.pack(side="bottom", pady=30)





window.mainloop()