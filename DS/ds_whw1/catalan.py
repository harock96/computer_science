def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper

def catalan(n):
    res = 0
    if (n == 0 or n == 1):
        return 1
    else:
        for i in range(n):
            res += catalan(i) * catalan(n-i-1)
        return res

catalan=memoize(catalan)

for i in range(100):
    print(catalan(i), end = " ")




