a=set()
s=set()


def expand(data):
    if(tuple(data) in s):
        return
    s.add(tuple(data))
    if len(data)==12:
        a.add(tuple(data))
        return
    for i in range(len(data)+1):
        if i==0:
            newData = [data[0]] + data
            expand(newData)
            continue
        if i==len(data):
            newData = data + [data[len(data)-1]]
            expand(newData)
            break
        newData = data[0:i] + [data[i-1]] + data[i:]
        expand(newData)
        newData = data[0:i] + [data[i]] + data[i:]
        expand(newData)
        
expand([1,26])
print(list(a))
