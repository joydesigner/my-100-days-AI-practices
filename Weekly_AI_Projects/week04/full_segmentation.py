# Dictionary; each word is followed by its frequency. The frequency is only an example and will not be used. You can also modify it yourself
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

# Sentence to be segmented
sentence = "经常有意见分歧"

# Implement the full segmentation function and output all the segmentation methods that can be segmented according to the dictionary
def all_cut(sentence, Dict):
    def dfs(remaining, path, result):
        # Recursion termination condition
        if not remaining:
            result.append(path)
            return
        # Limit the maximum segmentation length
        for i in range(1, len(remaining) + 1):
            word = remaining[:i] # Get the first i characters
            if word in Dict:
                dfs(remaining[i:], path + [word], result)  # Recursion
    result = []
    dfs(sentence, [], result)
    return result

# Call the full split function
target = all_cut(sentence, Dict)

# output
for t in target:
    print(t)

# Target output; order is not important
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]



