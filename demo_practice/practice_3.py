from pprint import pprint
from typing import Union


def binary_search(arr, key):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == key:
            if mid - 1 >= 0 and arr[mid-1] == key:
                high = mid - 1
            else:
                return mid
        elif arr[mid] > key:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def locate_card(cards, card_number):
    loc = binary_search(cards, card_number)
    if loc == -1:
        return 'Element not found'
    return loc


class Node:
    def __init__(self, val):
        self.val = val
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        node = Node(val)
        if self.head is None:
            self.head = node
        else:
            ptr = self.head
            while ptr.next is not None:
                ptr = ptr.next
            ptr.next = node
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
            ptr = ptr.next
        resp += 'NULL'
        return resp

    def __repr__(self):
        return self.__str__()


def reverse_linkedlist(ll: LinkedList):
    if ll.head is None:
        return None
    curr_node = ll.head
    prev_node = None

    while curr_node is not None:
        # save next node
        next_node = curr_node.next

        # update current node
        curr_node.next = prev_node

        # move pointers
        prev_node = curr_node
        curr_node = next_node

    ll.head = prev_node
    return ll.head


# Implement a binary tree using Python, and show its usage with some examples.
class TreeNode:
    def __init__(self, val):
        self.left = None
        self.val = val
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
    return inorder_traversal(node.left) + [node.val] + inorder_traversal(node.right)


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
    pass


def min_depth(node: Union[TreeNode, None]):
    pass


def tree_to_tuple(node: TreeNode):
    pass


def is_bst(node: Union[TreeNode, None]):
    if node is None:
        return True
    option1 = is_bst(node.left) and node.left is None or node.val >= node.left.val
    option2 = is_bst(node.right) and node.right is None or node.val <= node.right.val

    return option1 and option2


def bubble_sort(nums: list):
    nums_copy = nums.copy()
    for _ in range(len(nums_copy)-1):
        for i in range(len(nums_copy)-1):
            if nums_copy[i] > nums_copy[i+1]:
                nums_copy[i], nums_copy[i+1] = nums_copy[i+1], nums_copy[i]
    return nums_copy


def insertion_sort(nums: list):
    nums_copy = nums.copy()
    pos = 1
    while pos < len(nums_copy):
        pivot = nums_copy[pos]
        j = pos - 1
        while j >= 0 and nums_copy[j] > pivot:
            nums_copy[j+1] = nums_copy[j]
            j -= 1
        nums_copy[j+1] = pivot
        pos += 1
    return nums_copy


def merge_sort(nums: list):
    if len(nums) <= 1:
        return nums

    mid = len(nums) // 2

    left = nums[:mid]
    right = nums[mid:]

    sorted_left = merge_sort(left)
    sorted_right = merge_sort(right)

    return merge(sorted_left, sorted_right)


def merge(left: list, right: list):
    i = 0
    j = 0
    merged_list = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged_list.append(left[i])
            i += 1
        else:
            merged_list.append(right[j])
            j += 1

    merged_list.extend(left[i:])
    merged_list.extend(right[j:])
    return merged_list


def partition(arr: list, start, end):
    pivot = arr[end]
    l = start
    r = end - 1
    while r > l:
        if arr[l] <= pivot:
            l += 1
        elif arr[r] > pivot:
            r -= 1
        else:
            arr[l], arr[r] = arr[r], arr[l]
    if arr[l] > pivot:
        arr[l], arr[end] = arr[end], arr[l]
        return l
    else:
        return end


def quick_sort(arr: list, start, end):
    if start < end:
        pivot = partition(arr, start, end)
        quick_sort(arr, start, pivot - 1)
        quick_sort(arr, pivot + 1, end)
    return arr


def lcs(seq1, seq2, idx1=0, idx2=0):
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
    pass


# 0/1 Knapsack Problem
def max_profit(weights: list, profit: list, capacity: int, idx=0):
    pass


def max_profit_dp(weights, profit, capacity):
    pass


# tip1 => list of tips for waiter1
# tip2 => list of tips for waiter2
# capacity1 => number of tips waiter1 can take
# capacity2 => number of tips waiter2 can take
# total_orders => total orders received
def max_tip(tips1, tips2, capacity1, capacity2, idx=0):
    pass


def longest_increasing_subsequence(seq: list, idx, last_item):
    pass


def max_activity(start: list, finish: list, idx=0, next_start=0):
    pass


def main():
    # cards = [13, 13, 12, 12, 12, 12, 10, 9, 8, 7]  # cards in descending order
    # card_number = 12
    # print(locate_card(cards, card_number))

    # Linked List
    # ll = LinkedList().from_list([1,2,3,4])
    # print(ll)
    # reverse_linkedlist(ll)
    # print(ll)
    # tree_tuple = ((1, 3, None), 2, ((None, 3, 4), 5, (6, 7, 8)))
    # root_node = parse_tuple(tree_tuple)
    # display_tree(root_node)  # [1, 3, 2, 3, 4, 5, 6, 7, 8]
    # print(tree_height(root_node))  # 4
    # print(tree_size(root_node))  # 9
    # print(is_bst(root_node))  # False
    #
    # tree_tuple = ( 1, 3, (4, 5, 6) )
    # root_node = parse_tuple(tree_tuple)
    # display_tree(root_node)
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

    max_lcs, lcs_str = lcs('serendipitous', 'precipitation')
    print(max_lcs, lcs_str[::-1])
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

    # start = [10, 12, 20]
    # finish = [20, 25, 30]
    # start = [1, 3, 0, 5, 8, 5]
    # finish = [2, 4, 6, 7, 9, 9]
    # print(max_activity(start, finish))
    pass


if __name__ == '__main__':
    main()
