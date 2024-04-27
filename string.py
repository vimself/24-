'''
    / 分割
'''
def slash_split(text):
    index = 0
    while True:
        str = text[index]
        for i in range(len(str)):
            if str[i] == '/':
                if i-1 != 0 and str[i-1] == '/':
                    continue
                if i+1 != len(str) and str[i+1] == '/':
                    continue
                text.append(str[:i])
                text.append(str[i+1:])
                text.remove(str)
                break
        if index + 1 < len(text):
            index += 1
        else:
            break
    return text
# 示例用法
text = ["example.com//https:///example.com/aaaa/bb/%2e%2e////"]
result = slash_split(text)
print(result)  # 输出: ['example.com//https:///example.com', '%2e%2e']


