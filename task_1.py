def substring(s: str, t: str):
    s = s.strip()
    t = t.strip()
    n, m = len(s), len(t)
    t = ' ' + t
    nxt = [[-1] * 26 for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(26):
            nxt[i][j] = nxt[i + 1][j]
        nxt[i][ord(t[i + 1]) - ord('a')] = i + 1
    cur_pos = 0
    for i in range(n):
        c = ord(s[i]) - ord('a')
        if nxt[cur_pos][c] == -1:
            print('False')
            return
        cur_pos = nxt[cur_pos][c]
    print("True")

if __name__ == "__main__":
    s, t = input().strip().split()
    substring(s, t)
