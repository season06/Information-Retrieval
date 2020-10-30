if __name__ == '__main__':
    N = int(input())
    document, relevant = [], []
    for i in range(N):
        document.append(input())
        relevant.append(input())

    MAP = 0.0
    for index in range(N):
        relevant_dict = {} # 將 relevant 的資料存成 dictionary 以防有重複的 word
        precision = relevant[index].split()
        for j in precision:
            relevant_dict[j] = 0

        score = 0.0
        ans = 0
        doc = document[index].split(' ')

        for i in range(len(doc)):
            if doc[i] in relevant_dict: # 若文章中的字在 relevant doc 中有出現，則計算 score
                ans += 1
                score += ans / (i+1)  # 總共出現相關文章的次數 / 現在的文章數量

        MAP += score / len(precision) # 除以所有 relevant document 的數量

    MAP /= N # 除以 document 的數量
    MAP = round(MAP, 4)
    print(MAP)