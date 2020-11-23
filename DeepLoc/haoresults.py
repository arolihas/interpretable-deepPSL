import pandas as pd

file = 'haoresults.txt'


blank = ['']
cols = 5
with open(file) as f:
    allrows = []
    for line in f:
        line = line.split()
        if len(line) == 0:
            continue
        elif len(line) == 6:
            line[0] = ' '.join(line[:2])
            del line[1]
        elif len(line) == 4 and line[0] == 'f1-score':
            line.insert(0, blank[0])
        elif '-----' in line[0]:
            line[0] = ' '.join(line)
            line = [line[0]] + blank*4
        else:
            if len(line) < 5:
                line += (cols - len(line))*blank
            elif len(line) > 5:
                line = line[:6]
        # print(len(line), line)
        allrows.append(line)
        # _ = input("ENTER")
    df = pd.DataFrame(allrows)
    print(df)
    df.to_csv('haoresults.csv', index=False)