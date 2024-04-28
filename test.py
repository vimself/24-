arr = ['com','asd','asd']
res = list(set(arr))
print(res)
if 'com' in res:
    res.remove('com')
print(res)