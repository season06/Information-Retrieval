if __name__ == '__main__':
    N = int(input())
    document, relevant = [], []
    for i in range(N):
        document.append(input())
        relevant.append(input())

    MAP = 0.0
    for index in range(N):
        relevant_dict = {}
        precision = relevant[index].split()
        for j in precision:
            relevant_dict[j] = 0

        score = 0.0
        ans = 0
        doc = document[index].split(' ')

        for i in range(len(doc)):
            if doc[i] in relevant_dict:
                ans += 1
                score += ans / (i+1)

        MAP += score / len(precision)

    MAP /= N
    MAP = round(MAP, 4)
    print(MAP)