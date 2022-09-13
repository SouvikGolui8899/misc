from pprint import pprint
from typing import Union


def binary_search(arr, key):
    pass


def locate_card(cards, card_number):
    pass


class LinkedList:
    pass


def reverse_linkedlist(ll: LinkedList):
    pass


# Implement a binary tree using Python, and show its usage with some examples.
class TreeNode:
    pass

# tree_tuple = ((1,3,None), 2, ((None, 3, 4), 5, (6, 7, 8)))
def parse_tuple(data):
    pass


def inorder_traversal(node: TreeNode):
    pass


def display_tree(node: TreeNode):
    pass


def tree_height(node: TreeNode):
    pass


def tree_size(node: TreeNode):
    pass


def level_order_insert(arr: list):
    pass


def min_depth(node: Union[TreeNode, None]):
    pass


def tree_to_tuple(node: TreeNode):
    pass


def is_bst(node: Union[TreeNode, None]):
    pass


def bubble_sort(nums: list):
    pass


def insertion_sort(nums: list):
    pass


def merge_sort(nums: list):
    pass


def merge(left: list, right: list):
    pass


def partition(arr: list, start, end):
    pass


def quick_sort(arr: list, start, end):
    pass


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

    # start = [10, 12, 20]
    # finish = [20, 25, 30]
    # start = [1, 3, 0, 5, 8, 5]
    # finish = [2, 4, 6, 7, 9, 9]
    # print(max_activity(start, finish))
    pass


if __name__ == '__main__':
    main()
