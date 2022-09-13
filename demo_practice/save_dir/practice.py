from pprint import pprint
from typing import Union


def binary_search(arr, key):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == key:
            if mid - 1 >= 0 and arr[mid - 1] == key:
                high = mid - 1  # Change low = mid +1 for ascending order of elements
            else:
                return mid
        elif arr[mid] > key:  # Change the greater than or lesser than sigh accordingly
            low = mid + 1
        else:
            high = mid - 1
    return -1


def locate_card(cards, card_number):
    loc = binary_search(cards, card_number)
    return loc if loc != -1 else 'Element not found'


class Node:
    def __init__(self, val):
        self.val = val
        self.next_ptr = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        if self.head is None:
            self.head = Node(val)
        else:
            ptr = self.head
            while ptr.next_ptr is not None:
                ptr = ptr.next_ptr
            ptr.next_ptr = Node(val)
        return self

    def from_list(self, l: list):
        for item in l:
            self.append(item)
        return self

    def __str__(self):
        resp = ""
        ptr = self.head
        while ptr is not None:
            resp += f'{ptr.val} -> '
            ptr = ptr.next_ptr
        resp += 'NULL'
        return resp

    def __repr__(self):
        self.__str__()


def reverse_linkedlist(ll: LinkedList):
    if ll.head is None:
        return ll
    curr_ptr = ll.head
    prev_ptr = None
    while curr_ptr is not None:
        next_ptr = curr_ptr.next_ptr

        curr_ptr.next_ptr = prev_ptr

        prev_ptr = curr_ptr
        curr_ptr = next_ptr
    ll.head = prev_ptr


## Design a fast in-memory database using BST.
class User:
    def __init__(self, username: str):
        self.username = username
        # other attributes to be added

    def __repr__(self):
        return f"<User {self.username}>"

    def __str__(self):
        return self.__repr__()


class UserDatabase:
    def insert(self, user: User):
        pass

    def find(self, username: str):
        pass

    def update(self, user: User):
        pass

    def list_all(self):
        pass


# Implement a binary tree using Python, and show its usage with some examples.
class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


# tree_tuple = ((1,3,None), 2, ((None, 3, 4), 5, (6, 7, 8)))
def parse_tuple(data):
    if isinstance(data, tuple) and len(data) == 3:
        node = TreeNode(data[1])
        node.left = parse_tuple(data[0])
        node.right = parse_tuple(data[2])
    elif data is None:
        node = None
    else:
        node = TreeNode(data)
    return node


def inorder_traversal(node: TreeNode):
    if node is None:
        return []
    return (inorder_traversal(node.left)
            + [node.key] +
            inorder_traversal(node.right))


def display_tree(node: TreeNode):
    print(inorder_traversal(node))


def tree_height(node: TreeNode):
    if node is None:
        return 0
    return 1 + max(tree_height(node.left), tree_height(node.right))


def tree_size(node: TreeNode):
    if node is None:
        return 0
    return 1 + tree_size(node.left) + tree_size(node.right)


def level_order_insert(arr: list):
    pos = 1
    root = None
    while pos < len(arr):
        node = None
        if arr[pos] is not None:
            node = TreeNode(arr[pos])
            arr[pos] = None
            if (pos * 2) < len(arr):
                node.left = TreeNode(arr[pos * 2])
                # arr[pos * 2] = None
            else:
                node.left = None
            if (pos * 2) + 1 < len(arr):
                node.right = TreeNode(arr[(pos * 2) + 1])
                # arr[(pos * 2) + 1] = None
            else:
                node.right = None
        print(node)
        if node is not None:
            print(node.left.key, node.key, node.right.key)
        if pos == 1:
            root = node
        pos = pos + 1
    return root


def min_depth(node: Union[TreeNode, None]):
    if node is None:
        return 0
    return 1 + min(min_depth(node.left), min_depth(node.right))


def level_order_insertion(arr: list, pos=0):
    print(arr, pos)
    if pos >= len(arr):
        node = None
    elif arr[pos] is not None:
        node = TreeNode(arr[pos])
        node.left = level_order_insertion(arr, (pos * 2) +1)
        node.right = level_order_insertion(arr, (pos * 2) +2)
    elif arr[pos] is None:
        node = None
    else:
        node = TreeNode(arr[pos])
    return node


def tree_to_tuple(node: TreeNode):
    if node is None:
        return None
    if node.left is None and node.right is None:
        return (node.key)
    else:
        return (tree_to_tuple(node.left), node.key, tree_to_tuple(node.right))


def is_bst(node: Union[TreeNode, None]):
    if node is None:
        return True
    option1 = is_bst(node.left) and node.left is None or node.key >= node.left.key
    option2 = is_bst(node.right) and node.right is None or node.key <= node.right.key

    return option1 and option2


def bubble_sort(nums: list):
    nums_copy = nums[:]
    for _ in range(len(nums_copy) - 1):
        for i in range(len(nums_copy) - 1):
            if nums_copy[i] > nums_copy[i+1]:
                nums_copy[i], nums_copy[i+1] = nums_copy[i+1], nums_copy[i]
    return nums_copy


def insertion_sort(nums: list):
    nums_copy = nums[:]
    i = 1
    while i < len(nums_copy):
        j = i - 1
        curr_item = nums_copy[i]
        while j >= 0 and nums_copy[j] > curr_item:
            nums_copy[j+1] = nums_copy[j]
            j -= 1
        nums_copy[j+1] = curr_item
        i += 1
    return nums_copy


def merge_sort(nums: list):
    # If length of nums array is either 1 or 0 then return the array itself
    if len(nums) <= 1:
        return nums

    # Divide the array into two half
    mid = len(nums) // 2

    left = nums[:mid]
    right = nums[mid:]

    # Recurse on each half until the problem reduces to base case
    left_sorted = merge_sort(left)
    right_sorted = merge_sort(right)

    sorted_nums = merge(left_sorted, right_sorted)

    return sorted_nums


def merge(left: list, right: list):
    merged = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    # If one of the list has more elements than the other
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def partition(arr: list, start, end):
    l = start
    r = end - 1
    pivot_element = arr[end]
    while l < r:
        print(arr, l, r)
        if arr[l] <= pivot_element:
            l += 1
        elif arr[r] > pivot_element:
            r -= 1
        else:
            arr[l], arr[r] = arr[r], arr[l]

    if arr[l] > pivot_element:
        arr[l], arr[end] = arr[end], arr[l]
        return l
    else:
        return end


def quick_sort(arr: list, start, end):
    if start < end:
        pivot = partition(arr, start, end)
        quick_sort(arr, start, pivot-1)
        quick_sort(arr, pivot+1, end)
    return arr


def lcs(seq1, seq2, idx1=0, idx2=0):
    # Create two pointers at index 0. idx1 = 0 and idx2 = 0. Our recursive function should compute
    # substring of seq1[idx1:] and seq2[idx2:].
    # If seq1[idx1] == seq2[idx2] then seq1[idx1] or seq2[idx2] belongs to subsequence of seq1[idx1:]
    # and seq2[idx2:].
    # If seq1[idx1] != seq2[idx2] then LCS of seq1[idx1:] and seq2[idx2:] is the longer among
    # the LCS of seq1[idx1+1:], seq2[idx2:] and seq1[idx1:], seq2[idx2+1:]
    if idx1 == len(seq1) or idx2 == len(seq2):
        return 0
    if seq1[idx1] == seq2[idx2]:
        return 1+lcs(seq1, seq2, idx1+1, idx2+1)
    else:
        return max(lcs(seq1, seq2, idx1+1, idx2), lcs(seq1, seq2, idx1, idx2+1))


def lcs_print(seq1, seq2, idx1=0, idx2=0):
    if idx1 == len(seq1) or idx2 == len(seq2):
        return 0, ''
    elif seq1[idx1] == seq2[idx2]:
        max_lcs, lcs_str = lcs(seq1, seq2, idx1+1, idx2+1)
        return 1 + max_lcs, f'{lcs_str}{seq1[idx1]}'

    else:
        option1_lcs, lcs_str1 = lcs(seq1, seq2, idx1+1, idx2)
        option2_lcs, lcs_str2 = lcs(seq1, seq2, idx1, idx2+1)
        if option1_lcs > option2_lcs:
            return option1_lcs, f'{lcs_str1}'
        else:
            return option2_lcs, f'{lcs_str2}'


def lcs_dp(seq1, seq2):
    table = [[0 for _ in range(len(seq1)+1)] for _ in range(len(seq2)+1)]
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if seq1[i] == seq2[j]:
                table[i+1][j+1] = 1 + table[i][j]
            else:
                table[i+1][j+1] = max(table[i+1][j], table[i][j+1])
    i = len(seq1)
    j = len(seq2)
    lcs_str = ""
    while i > 0 and j > 0:
        # pprint(table)
        # print(i, j)
        if seq1[i-1] == seq2[j-1]:
            lcs_str += seq1[i-1]
            i -= 1
            j -= 1
        elif table[i-1][j] > table[i][j-1]:
            i -= 1
        else:
            j -= 1
    return table[-1][-1], lcs_str[::-1]


# 0/1 Knapsack Problem
def max_profit(weights: list, profit: list, capacity: int, idx=0):
    if len(weights[idx:]) == 0:
        return 0
    if weights[idx] > capacity:
        return max_profit(weights, profit, capacity, idx + 1)
    else:
        # We don't include the current weight in hand
        option1 = max_profit(weights, profit, capacity, idx + 1)
        # We include the current weight in hand
        option2 = profit[idx] + max_profit(weights, profit, capacity - weights[idx], idx + 1)
        return max(option1, option2)


def max_profit_dp(weights, profit, capacity):
    table = [[0 for _ in range(capacity+1)] for _ in range(len(weights)+1)]
    for idx in range(len(weights)):
        for c in range(capacity+1):
            if weights[idx] > c:
                table[idx+1][c] = table[idx][c]
            else:
                options1 = table[idx][c]
                options2 = profit[idx] + table[idx][c-weights[idx]]
                table[idx+1][c] = max(options1, options2)
    return table[-1][-1]


# tip1 => list of tips for waiter1
# tip2 => list of tips for waiter2
# capacity1 => number of tips waiter1 can take
# capacity2 => number of tips waiter2 can take
# total_orders => total orders received
def max_tip(tips1, tips2, capacity1, capacity2, idx=0):
    if len(tips1[idx:]) == 0 or len(tips2[idx:]) == 0:
        return 0
    if capacity1 == 0 and capacity2 == 0:
        return max_tip(tips1, tips2, capacity1, capacity2, idx+1)
    elif capacity1 > 0 and capacity2 == 0:
        return tips1[idx] + max_tip(tips1, tips2, capacity1 - 1, capacity2, idx+1)
    elif capacity1 == 0 and capacity2 > 0:
        return tips2[idx] + max_tip(tips1, tips2, capacity1, capacity2 - 1, idx + 1)
    else:
        return max(tips1[idx] + max_tip(tips1, tips2, capacity1 - 1, capacity2, idx+1), tips2[idx] + max_tip(tips1, tips2, capacity1, capacity2 - 1, idx + 1))

    # If capacity of waiter1 and capacity of waiter2 is full then the max_tip is same as the previous
    # If capacity of waiter1 and waiter2 is not full  then max(tip_from_waiter1, tip_from_waiter2)

def longest_increasing_subsequence(seq: list, idx, last_item):
    # If seq[idx] > all(seq[idx:]) then seq[idx] is not in LIS of seq[idx:]
    # Else seq[idx] belongs to LIS of seq[idx:]
    print(idx)
    if idx < 0:
        return 0, []
    if seq[idx] > seq[last_item]:
        return longest_increasing_subsequence(seq, idx-1, last_item)
    else:
        option1, lis1 = longest_increasing_subsequence(seq, idx-1, idx)
        option2, lis2 = longest_increasing_subsequence(seq, idx-1, last_item)
        max_lis = max(1+option1, option2)
        res_lis = []
        if max_lis == 1+option1:
            res_lis = lis1 + [seq[idx]]
        return max_lis, res_lis


def max_activity(start: list, finish: list, idx=0, next_start=0):
    if idx == len(start):
        return 0, []
    # If an activity is included then
    if next_start > start[idx]:
        return max_activity(start, finish, idx+1, next_start)
    else:
        option1, activity_list1 = max_activity(start, finish, idx+1, next_start)
        option2, activity_list2 = max_activity(start, finish, idx+1, finish[idx])
        maximum_activity = max(option1, 1+option2)
        if maximum_activity == 1+option2:
            res_list = activity_list2 + [idx]
        else:
            res_list = activity_list1
        return maximum_activity, res_list


def main():
    # cards = [13, 13, 12, 12, 12, 12, 10, 9, 8, 7]  # cards in descending order
    # card_number = 13
    # print(locate_card(cards, card_number))

    # Linked List
    # ll = LinkedList().from_list([1,2,3,4])
    # print(ll)
    # reverse_linkedlist(ll)
    # print(ll)
    # tree_tuple = ((1, 3, None), 2, ((None, 3, 4), 5, (6, 7, 8)))
    # root_node = parse_tuple(tree_tuple)
    # display_tree(root_node)
    # print(tree_height(root_node))
    # print(tree_size(root_node))

    # root_node = level_order_insertion([3,9,20,None,None,15,7])
    # print(root_node)
    # display_tree(root_node)
    # print(tree_to_tuple(root_node))
    # print(is_bst(root_node))
    #
    # root_node = level_order_insertion([9, 3, 20, None, None, 15, 21])
    # print(tree_to_tuple(root_node))
    # print(is_bst(root_node))

    # root_node = level_order_insertion([2,None,3,None,4,None,5,None,6])
    # print(tree_to_tuple(root_node))
    # print(min_depth(root_node))
    # print(bubble_sort([9, 8, 7, 14, 15, 30, 3, 2, 1]))
    # print(insertion_sort([9, 8, 7, 14, 15, 30, 3, 2, 1]))
    # print(merge_sort([9, 8, 7, 14, 15, 30, 3, 2, 1]))

    # nums = [9, 8, 7, 14, 15, 30, 3, 2, 1]
    # print(quick_sort(nums[:], 0, len(nums) -1))

    # print(lcs('serendipitous', 'precipitation'))
    # print(lcs_dp('serendipitous', 'precipitation'))

    # profits = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
    # weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
    # capacity = 165
    # print(max_profit(weights, profits, capacity))
    # print(max_profit_dp(weights, profits, capacity))

    # N = 5
    # X = 3
    # Y = 3
    # A = [1, 2, 3, 4, 5]
    # B = [5, 4, 3, 2, 1]
    # print(max_tip(A, B, X, Y))
    #
    # N = 7
    # X = 3
    # Y = 4
    # A = [8, 7, 15, 19, 16, 16, 18]
    # B = [1, 7, 15, 11, 12, 31, 9]
    # print(max_tip(A, B, X, Y))

    # seq = [3, 10, 2, 1, 20]
    # seq = [50, 3, 10, 7, 40, 80]
    # print(longest_increasing_subsequence(seq, len(seq) - 1, len(seq) - 1))

    start = [10, 12, 20]
    finish = [20, 25, 30]
    # start = [1, 3, 0, 5, 8, 5]
    # finish = [2, 4, 6, 7, 9, 9]
    print(max_activity(start, finish))


if __name__ == '__main__':
    main()
