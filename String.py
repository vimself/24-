'''
    / 分割
'''
def slash_split(text):
    index = 0
    while True:
        str = text[index]
        for i in range(len(str)):
            if str[i] == '/':
                if i-1 == -1:
                    text.append(str[i+1:])
                    text.remove(str)
                    break
                if i+1 == len(str) and str[i] == '/' and str[i-1] != '/':
                    text.append(str[:i])
                    text.remove(str)
                    break
                if str[i-1] == '/' or str[i+1] == '/':
                    continue
                if i-2 >= 0 and str[i-1] == '.' and str[i-2] == '.':
                    continue
                if str[i+1] == '&':
                    continue
                if str[i+1] == '>':
                    continue
                if str[i+1] == '?' and i+2 < len(str):
                    text.append(str[:i])
                    text.append(str[i+2:])
                    text.remove(str)
                    break
                text.append(str[:i])
                text.append(str[i+1:])
                text.remove(str)
                break
        if index + 1 < len(text):
            index += 1
        else:
            break
    return text

'''
    ? 分割
'''
def question_split(text):
    index = 0
    while True:
        str = text[index]
        for i in range(len(str)):
            if str[i] == '?':
                if i - 1 == -1:
                    text.append(str[i + 1:])
                    text.remove(str)
                    break
                if i + 1 == len(str) and str[i] == '?' and str[i - 1] != '?':
                    text.append(str[:i])
                    text.remove(str)
                    break
                if str[i - 1] == '?' or str[i + 1] == '?':
                    continue
                text.append(str[:i])
                text.append(str[i + 1:])
                text.remove(str)
                break
        if index + 1 < len(text):
            index += 1
        else:
            break
    return text

'''
    & 分割
'''
def address_split(text):
    index = 0
    while True:
        str = text[index]
        for i in range(len(str)):
            if str[i] == '&':
                if i - 1 == -1:
                    text.append(str[i + 1:])
                    text.remove(str)
                    break
                if i + 1 == len(str) and str[i] == '&' and str[i - 1] != '&':
                    text.append(str[:i])
                    text.remove(str)
                    break
                if str[i - 1] == '&' or str[i + 1] == '&':
                    continue
                if str[i-1] == '/':
                    text.append(str[:i-1])
                    text.append(str[i+1:])
                    text.remove(str)
                    break
                text.append(str[:i])
                text.append(str[i + 1:])
                text.remove(str)
                break
        if index + 1 < len(text):
            index += 1
        else:
            break
    return text

'''
    <br/> 分割
'''
def br_split(text):
    res = []
    for i in range(len(text)):
        arr = text[i].split("<br/>")
        res = res + arr
    return res

def amp_split(text):
    res = []
    for i in range(len(text)):
        arr = text[i].split("&amp;")
        res = res + arr
    return res


def separation(arr):
    result1 = br_split(arr)
    result2 = amp_split(result1)
    result3 = slash_split(result2)
    result4 = question_split(result3)
    result5 = address_split(result4)
    return result5
print(separation(['example.com//https:///example.com']))

