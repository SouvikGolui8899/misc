from collections import defaultdict


def two_sum(arr: list, sum: int):
    hash_map = {}
    i = 0
    save_res = {}
    while i < len(arr):
        x = arr[i]
        y = sum - arr[i]
        j = hash_map.get(y, None)
        if j is not None and (y, x) not in save_res:
            save_res[(i, j)] = (x, y)
        else:
            hash_map[arr[i]] = i
        i += 1
    return save_res


def max_profit_from_stock(prices: list):
    buy = 0
    sell = 1
    profit = 0
    while sell < len(prices):
        if prices[buy] > prices[sell]:
            buy = sell  # Move your buy (left) pointer all the way upto the sell because you found a really small price than the previous
        elif profit < prices[sell] - prices[buy]:
            profit = prices[sell] - prices[buy]
        sell += 1
    return profit


def contains_duplicate(nums: list):
    hash_set = set()
    for item in nums:
        if item in hash_set:
            return True
        else:
            hash_set.add(item)
    return False


def is_anagram(s: str, t: str):
    if len(s) != len(t):
        return False
    hash_map1 = {}
    hash_map2 = {}
    i = 0
    while i < len(s):
        if s[i] not in hash_map1:
            hash_map1[s[i]] = 1
        else:
            hash_map1[s[i]] += 1
        if t[i] not in hash_map2:
            hash_map2[t[i]] = 1
        else:
            hash_map2[t[i]] += 1
        i += 1
    return hash_map1 == hash_map2


def group_anagrams(strs: list):
    res = defaultdict(list)
    for s in strs:
        count = [0 for _ in range(26)]
        for c in s:
            count[ord(c) - ord('a')] += 1
        res[tuple(count)].append(s)
    return list(res.values())


def top_k_frequent(nums: list, k: int):
    hash_map = {}
    for item in nums:
        hash_map[item] = 1 + hash_map.get(item, 0)
    res = []
    for item in hash_map:
        if hash_map[item] >= k:
            res.append(item)
    # print(hash_map)
    return res


def product_except_self(nums: list):
    prefix_arr = [0] * (len(nums) + 1)
    postfix_arr = [0] * (len(nums) + 1)

    prefix_arr[0] = 1
    postfix_arr[-1] = 1

    pos = 1
    while pos < len(prefix_arr):
        print(pos)
        prefix_arr[pos] = nums[pos - 1] * prefix_arr[pos - 1]
        pos += 1
    pos = len(nums) - 1
    while pos >= 0:
        print(pos)
        postfix_arr[pos] = nums[pos] * postfix_arr[pos + 1]
        pos -= 1

    print(prefix_arr, postfix_arr)

    res = []

    pos = 1
    while pos < len(prefix_arr):
        res.append(prefix_arr[pos-1] * postfix_arr[pos])
        pos += 1
    return res


def encode(strs: list):
    res = ""
    for item in strs:
        res += f"{len(item)}#{item}"
    return res


def decode(encodes_str):
    res = []
    pos = 0
    while pos < len(encodes_str):
        str_len = ""
        while encodes_str[pos] != '#':
            str_len += encodes_str[pos]
            pos += 1
        str_len = int(str_len)
        pos += 1
        start_pos = pos
        out_str = ""
        while pos < start_pos + str_len:
            out_str += encodes_str[pos]
            pos += 1
        res.append(out_str)
    return res


def is_palindrome(s: str):
    l = 0
    r = len(s) - 1
    while l < r:
        print(l, r, s[l], s[r])
        if ord(s[l].lower()) < ord('a') or ord(s[r].lower()) > ord('z'):
            l += 1
        elif ord(s[r].lower()) < ord('a') or ord(s[r].lower()) > ord('z'):
            r -= 1
        elif s[l].lower() != s[r].lower():
            return False
        else:
            l += 1
            r -= 1
    return True


def unique_paths(m: int, n: int):
    # print(m,n)
    if m == 1 or n == 1:
        return 1
    else:
        option1 = unique_paths(m-1, n)
        option2 = unique_paths(m, n-1)
    return option1 + option2


def valid_parentheses(s: str):
    pos = 0
    stack = []
    parentheses_map = {
        ')': '(',
        '}': '{',
        ']': '['
    }
    while pos < len(s):
        # print(stack)
        if s[pos] in parentheses_map.values():
            stack.append(s[pos])
        elif s[pos] in parentheses_map.keys():
            if stack == []:
                return False
            if parentheses_map.get(s[pos]) == stack[-1]:
                stack.pop()
        pos += 1
    if stack == []:
        return True
    else:
        return False


def climb_stairs(n: int):
    steps = [0 for _ in range(n+1)]
    steps[0] = 1
    for i in range(1,n+1):
        ways = 0
        if i-1 >= 0:
            ways += steps[i-1]
        if i-2 >= 0:
            ways += steps[i-2]
        steps[i] = ways

    # if n == 0:
    #     return 1
    # option1 = 0
    # option2 = 0
    # if n-1 >= 0:
    #     option1 = climb_stairs(n-1)
    # if n-2 >= 0:
    #     option2 = climb_stairs(n-2)
    # return option1 + option2
    return steps[-1]



def main():
    # print(two_sum([1, 4, 2, 3, 0, 5], 5))
    # print(max_profit_from_stock([7,1,5,3,6,4]))
    # print(max_profit_from_stock([7,6,4,3,1]))
    # print(max_profit_from_stock([2,1,2,1,0,1,2]) == 2)

    # print(contains_duplicate([1,2,3,1]))
    # print(contains_duplicate([1,2,3,4]))

    # print(is_anagram('anagram', 'nagaram'))
    # print(is_anagram('rat', 'car'))

    # print(group_anagrams(["eat","tea","tan","ate","nat","bat"]))
    # print(group_anagrams([""]))
    # print(group_anagrams(["a"]))

    # print(top_k_frequent([1,1,1,2,2,3], 2))
    # print(top_k_frequent([1], 1))

    # print(product_except_self([1,2,3,4]))

    # str_list = ["leet", "code", "love", "you"]
    # encoded_str = encode(str_list)
    # print(encoded_str)
    # print(decode(encoded_str))

    # print(is_palindrome("A man, a plan, a canal: Panama"))
    # print(is_palindrome("race a car"))
    # print(unique_paths(3, 2))
    # print(unique_paths(2, 2))
    # print(valid_parentheses('()[]{}'))
    # print(valid_parentheses('(]'))
    # print(valid_parentheses('(a+(b*c))'))
    # print(valid_parentheses('(a+b*c))'))
    print(climb_stairs(2))
    print(climb_stairs(38))
    pass


if __name__ == '__main__':
    main()
