from pprint import pprint
from typing import Union


def binary_search(arr, key):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == key:
            if mid - 1 > 0 and arr[mid] == arr[mid - 1]:
                high = mid - 1
            else:
                return mid
        elif arr[mid] > key:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def locate_card(cards, card_number):
    found = binary_search(cards, card_number)
    return found if found != -1 else 'Not Found'


class Node:
    def __init__(self, val):
        self.val = val
        self.next_ptr = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        node = Node(val)
        if self.head is None:
            self.head = node
        else:
            ptr = self.head
            while ptr.next_ptr is not None:
                ptr = ptr.next_ptr
            ptr.next_ptr = node
        return self

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        resp = ""
        ptr = self.head
        while ptr is not None:
            resp += f"{ptr.val} ->"
            ptr = ptr.next_ptr
        resp += 'NULL'
        return resp

    def from_list(self, arr: list):
        for item in arr:
            self.append(item)
        return self


def reverse_linkedlist(ll: LinkedList):
    if ll.head is None:
        return None
    curr_node = ll.head
    prev_node = None
    while curr_node is not None:
        next_node = curr_node.next_ptr

        curr_node.next_ptr = prev_node

        prev_node = curr_node
        curr_node = next_node

    ll.head = prev_node



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
    pass


# Implement a binary tree using Python, and show its usage with some examples.
class TreeNode:
    def __init__(self, val):
        self.val = val
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


def level_order_insertion(arr: list, pos=0):
    pass


def tree_to_tuple(node: TreeNode):
    if node is None:
        return None
    if node.left is None and node.right is None:
        return (node.val)
    else:
        return (tree_to_tuple(node.left), node.val, tree_to_tuple(node.right))


def is_bst(node: Union[TreeNode, None]):
    # Recursive definition
    # If empty tree is bst
    # If the left subtree is bst and right subtree is bst then its a bst
    if node is None:
        return True
    if (node.left is not None and node.left.val > node.val) or (node.right is not None and node.val > node.right.val):
        return False
    is_bst_left = is_bst(node.left)
    is_bst_right = is_bst(node.right)
    return is_bst_left and is_bst_right


def bubble_sort(nums: list):
    nums_copy = nums[:]
    for _ in range(len(nums_copy) - 1):
        for j in range(len(nums_copy) - 1):
            if nums_copy[j] > nums_copy[j+1]:
                nums_copy[j], nums_copy[j+1] = nums_copy[j+1], nums_copy[j]
    return nums_copy


def insertion_sort(nums: list):
    nums_copy = nums[:]
    i = 1
    while i < len(nums_copy):
        j = i - 1
        curr_element = nums_copy[i]
        while j >= 0 and nums_copy[j] > curr_element:
            nums_copy[j+1] = nums_copy[j]
            j -= 1
        nums_copy[j+1] = curr_element
        i += 1
    return nums_copy


def merge_sort(nums: list):
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    left = nums[:mid]
    right = nums[mid:]

    left_sorted = merge_sort(left)
    right_sorted = merge_sort(right)

    return merge(left_sorted, right_sorted)


def merge(left: list, right: list):
    final = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            final.append(left[i])
            i += 1
        else:
            final.append(right[j])
            j += 1
    final.extend(left[i:])
    final.extend(right[j:])
    return final


def partition(arr: list, start, end):
    pivot = arr[end]
    l = start
    r = end - 1
    while l < r:
        if arr[l] < pivot:
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
        quick_sort(arr, start, pivot-1)
        quick_sort(arr, pivot + 1, end)
    return arr


def lcs(seq1, seq2, idx1=0, idx2=0):
    pass


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
    # display_tree(root_node)
    # print(tree_height(root_node))
    # print(tree_size(root_node))
    # print(is_bst(root_node))
    #
    # tree_tuple = ((1, 1, None), 2, (3, 5, 6))
    # root_node = parse_tuple(tree_tuple)
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
    # print('Bubble Sort', bubble_sort([9, 8, 7, 14, 15, 30, 3, 2, 1]))
    # print('Insertion Sort',insertion_sort([9, 8, 7, 14, 15, 30, 3, 2, 1]))
    # print('Merge Sort',merge_sort([9, 8, 7, 14, 15, 30, 3, 2, 1]))
    #
    # nums = [9, 8, 7, 14, 15, 30, 3, 2, 1]
    # print('Quick Sort', quick_sort(nums[:], 0, len(nums) -1))
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

    # start = [10, 12, 20]
    # finish = [20, 25, 30]
    # start = [1, 3, 0, 5, 8, 5]
    # finish = [2, 4, 6, 7, 9, 9]
    # print(max_activity(start, finish))
    pass


if __name__ == '__main__':
    main()
