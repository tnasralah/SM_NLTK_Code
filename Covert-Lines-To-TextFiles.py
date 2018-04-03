
li_lines=[]
filepath="data/discussions - ALL-Technology SubCatigories.txt"
cnt=0
with open(filepath) as f:
    for line in f:
        li_lines.append(line)
        cnt = cnt+ 1
        print (line)
print (cnt)

for i in range(len(li_lines)):
    file_name= 'd'+str(i)+'.txt'
    d= li_lines[i]
    if d.strip()!='':
        with open("data/ALLTech-SubCates/"+file_name, "a") as f:
            f.write(li_lines[i])
            f.close()
    else:
        print (i)
