import os 
def fix(label):
    with open(label,"r") as f:
        lines=f.readlines()
    new=[]
    for line in lines:
        parts=line.strip().split()
        if parts and parts[0]=='0':
            parts[0]='8'
        new.append(" ".join(parts)+"\n")
    with open(label,"w") as f:
        f.writelines(new)

def patch(label_dir):
    for root,_, files in os.walk(label_dir):
        for f in files:
            if f.endswith(".txt"):
                label=os.path.join(root,f)
                fix(label)
label_folder=r"data\\smoke"
patch(label_folder)