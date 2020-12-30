# Skeleton file for HW5 - Winter 2021 - extended intro to CS

# Add your implementation to this file

# You may add other utility functions to this file,
# but you may NOT change the signature of the existing ones.

# Change the name of the file to include your ID number (hw5_ID.py).

# Enter all IDs of participating students as strings, separated by commas.

# For example: SUBMISSION_IDS = ["123456", "987654"] if submitted in a pair or SUBMISSION_IDS = ["123456"] if submitted alone.

SUBMISSION_IDS = ["209729524", "211964515"]


############
# QUESTION 1
############

def printree(t, bykey=True):
    """Print a textual representation of t
    bykey=True: show keys instead of values"""
    # for row in trepr(t, bykey):
    #        print(row)
    return trepr(t, bykey)


def trepr(t, bykey=False):
    """Return a list of textual representations of the levels in t
    bykey=True: show keys instead of values"""
    if t == None:
        return ["#"]

    thistr = str(t.key) if bykey else str(t.val)

    return conc(trepr(t.left, bykey), thistr, trepr(t.right, bykey))


def conc(left, root, right):
    """Return a concatenation of textual represantations of
    a root node, its left node, and its right node
    root is a string, and left and right are lists of strings"""

    lwid = len(left[-1])
    rwid = len(right[-1])
    rootwid = len(root)

    result = [(lwid + 1) * " " + root + (rwid + 1) * " "]

    ls = leftspace(left[0])
    rs = rightspace(right[0])
    result.append(ls * " " + (lwid - ls) * "_" + "/" + rootwid * " " + "|" + rs * "_" + (rwid - rs) * " ")

    for i in range(max(len(left), len(right))):
        row = ""
        if i < len(left):
            row += left[i]
        else:
            row += lwid * " "

        row += (rootwid + 2) * " "

        if i < len(right):
            row += right[i]
        else:
            row += rwid * " "

        result.append(row)

    return result


def leftspace(row):
    """helper for conc"""
    # row is the first row of a left node
    # returns the index of where the second whitespace starts
    i = len(row) - 1
    while row[i] == " ":
        i -= 1
    return i + 1


def rightspace(row):
    """helper for conc"""
    # row is the first row of a right node
    # returns the index of where the first whitespace ends
    i = 0
    while row[i] == " ":
        i += 1
    return i


class Tree_node():
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        return "(" + str(self.key) + ":" + str(self.val) + ")"


class Binary_search_tree():

    def __init__(self):
        self.root = None

    def __repr__(self):  # no need to understand the implementation of this one
        out = ""
        for row in printree(self.root):  # need printree.py file
            out = out + row + "\n"
        return out

    def lookup(self, key):
        ''' return node with key, uses recursion '''

        def lookup_rec(node, key):
            if node == None:
                return None
            elif key == node.key:
                return node
            elif key < node.key:
                return lookup_rec(node.left, key)
            else:
                return lookup_rec(node.right, key)

        return lookup_rec(self.root, key)

    def insert(self, key, val):
        ''' insert node with key,val into tree, uses recursion '''

        def insert_rec(node, key, val):
            if key == node.key:
                node.val = val  # update the val for this key
            elif key < node.key:
                if node.left == None:
                    node.left = Tree_node(key, val)
                else:
                    insert_rec(node.left, key, val)
            else:  # key > node.key:
                if node.right == None:
                    node.right = Tree_node(key, val)
                else:
                    insert_rec(node.right, key, val)
            return

        if self.root == None:  # empty tree
            self.root = Tree_node(key, val)
        else:
            insert_rec(self.root, key, val)

    def diam(self):
        def diam_node(node):
            if node is None:
                return 0, 0
            l_depth, l_diam = diam_node(node.left)
            r_depth, r_diam = diam_node(node.right)
            return 1 + max(r_depth, l_depth), max(1 + r_depth + l_depth, l_diam, r_diam)
        return diam_node(self.root)[1]

    def is_min_heap(self):
        def is_min_heap_node(node):
            if node is None:
                return True
            if node.left is not None and node.left.val < node.val:
                return False
            if node.right is not None and node.right.val < node.val:
                return False
            return is_min_heap_node(node.left) and is_min_heap_node(node.right)
        return is_min_heap_node(self.root)


############
# QUESTION 2
############

# Part A
class PNode():

    def __init__(self, val, p, next=None):
        self.value = val
        self.next = next
        self.priority = p

    def __repr__(self):
        return f"{self.value},{self.priority} ({id(self)})"
    # This shows pointers as well for educational purposes


class PQueue():

    def __init__(self, vals=None, ps=None):
        self.next = None
        self.len = 0
        if vals != None:
            for val, p in zip(vals, ps):
                self.insert(val, p)

    def __len__(self):
        return self.len

    def __repr__(self):
        out = ""
        p = self.next
        while p != None:
            out += str(p) + ", "  # str(p) envokes __repr__ of class PNode
            p = p.next
        return "[" + out[:-2] + "]"

    def pull(self):
        retVal = self.next
        if retVal is None:
            return None, None
        self.len -= 1
        self.next = self.next.next
        return retVal.value, retVal.priority

    def insert(self, val, p):
        curr = self.next
        self.len += 1
        if curr is None:
            self.next = PNode(val, p)
            return
        if curr.priority < p:
            self.next = PNode(val, p)
            self.next.next = curr
            return
        while curr.next is not None and curr.next.priority >= p:
            curr = curr.next
        new_node = PNode(val, p)
        new_node.next = curr.next
        curr.next = new_node



#  Part B
class Node():
    def __init__(self, val):
        self.value = val
        self.next = None

    def __repr__(self):
        # return str(self.value)
        # This shows pointers as well for educational purposes:
        return "(" + str(self.value) + ", next: " + str(id(self.next)) + ")"


class Linked_list():
    def __init__(self, seq=None):
        self.next = None
        self.len = 0
        if seq != None:
            for x in seq[::-1]:
                self.add_at_start(x)

    def __repr__(self):
        out = ""
        p = self.next
        while p != None:
            out += str(p) + ", "  # str(p) envokes __repr__ of class Node
            p = p.next
        return "[" + out[:-2] + "]"

    def __len__(self):
        ''' called when using Python's len() '''
        return self.len

    def add_at_start(self, val):
        ''' add node with value val at the list head '''
        p = self
        tmp = p.next
        p.next = Node(val)
        p.next.next = tmp
        self.len += 1

    def reverse_start_end(self, k):
        assert 0 <= k <= self.len / 2
        first_node = self.next
        next_node = first_node.next
        curr = first_node
        prev = None
        for i in range(k):
            curr.next = prev
            prev = curr
            curr = next_node
            next_node = next_node.next
        first_node.next = curr
        self.next = prev
        for i in range(self.len - 2*k):
            prev = curr
            curr = next_node
            next_node = next_node.next
        first_node = prev
        prev = None
        for i in range(k-1):
            curr.next = prev
            prev = curr
            curr = next_node
            next_node = next_node.next
        curr.next = prev
        first_node.next = curr





############
# QUESTION 4
############

# a
def power_new(a, b):
    """ computes a**b using iterated squaring

        assumes b is nonnegative integer """

    result = 1

    b_bin = bin(b)[2:]

    reverse_b_bin = b_bin[::-1]

    for bit in reverse_b_bin:
        pass
#    _____________________________________
#
#    _____________________________________
#
#   _____________________________________

    return result


# b
def power_with_base(a, b, base=2):
    """ computes a**b using iterated squaring

        with divisor base >= 2

        assumes b is nonnegative integer """
    assert base >= 2
    result = 1

    while b > 0:

        x = 1

#        residual = __________________________

#        for i in range(residual):
#            x *= a

#        _____________________________________

#        for i in range(__________________):
#            x *= a

#        _____________________________________

        b = b // base

    return result


############
# QUESTION 5
############
# a
def prefix_suffix_overlap(lst, k):
    pass  # replace this with your code


# c
#########################################
### Dict class ###
#########################################

class Dict:
    def __init__(self, m, hash_func=hash):
        """ initial hash table, m empty entries """
        self.table = [[] for i in range(m)]
        self.hash_mod = lambda x: hash_func(x) % m

    def __repr__(self):
        L = [self.table[i] for i in range(len(self.table))]
        return "".join([str(i) + " " + str(L[i]) + "\n" for i in range(len(self.table))])

    def insert(self, key, value):
        """ insert key,value into table
            Allow repetitions of keys """
        i = self.hash_mod(key)  # hash on key only
        item = [key, value]  # pack into one item
        self.table[i].append(item)

    def find(self, key):
        """ returns ALL values of key as a list, empty list if none """
        pass  # replace this with your code


#########################################
### End Dict class ###
#########################################

# d
def prefix_suffix_overlap_hash1(lst, k):
    pass  # replace this with your code


# f
def prefix_suffix_overlap_hash2(lst, k):
    pass  # replace this with your code


########
# Tester
########

def test():
    # Question 1 - a

    t2 = Binary_search_tree()
    t2.insert('c', 10)
    t2.insert('a', 10)
    t2.insert('b', 10)
    t2.insert('g', 10)
    t2.insert('e', 10)
    t2.insert('d', 10)
    t2.insert('f', 10)
    t2.insert('h', 10)
    if t2.diam() != 6:
        print("error in Binary_search_tree.diam")

    t3 = Binary_search_tree()
    t3.insert('c', 1)
    t3.insert('g', 3)
    t3.insert('e', 5)
    t3.insert('d', 7)
    t3.insert('f', 8)
    t3.insert('h', 6)
    t3.insert('z', 6)
    if t3.diam() != 5:
        print("error in Binary_search_tree.diam")

    # Question 1 - b
    """ Construct below binary tree
               1
             /   \
            /     \
           2       3
          / \     / \
         /   \   /   \
        17   19 36    7
    """
    t1 = Binary_search_tree()
    t1.insert('d', 1)
    t1.insert('b', 2)
    t1.insert('a', 17)
    t1.insert('c', 19)
    t1.insert('f', 3)
    t1.insert('e', 36)
    t1.insert('g', 7)

    if not t1.is_min_heap():
        print("Error in is_min_heap")

    # Question 2 - a
    pq = PQueue("abc", [2, 1, 3])
    if pq.pull()[0] != "c":
        print("error in PQueue.pull")
    if pq.pull()[0] != "a":
        print("error in PQueue.pull")

    # Question 2 - b
    q = Linked_list("abcde")
    q.reverse_start_end(2)
    q_str = []
    p = q.next
    while p != None:
        q_str += [p.value]
        p = p.next
    q_str = "".join(q_str)
    if q_str != "baced":
        print("error in reverse_start_end")

    # Question 4
    if power_new(2, 3) != 8:
        print("error in power_new()")

    if power_with_base(2, 3, 3) != 8:
        print("error in power_with_base()")

    # Question 5
    s0 = "a" * 100
    s1 = "b" * 40 + "a" * 60
    s2 = "c" * 50 + "b" * 40 + "a" * 10
    lst = [s0, s1, s2]
    k = 50
    if prefix_suffix_overlap(lst, k) != [(0, 1), (1, 2)] and \
            prefix_suffix_overlap(lst, k) != [(1, 2), (0, 1)]:
        print("error in prefix_suffix_overlap")
    if prefix_suffix_overlap_hash1(lst, k) != [(0, 1), (1, 2)] and \
            prefix_suffix_overlap_hash1(lst, k) != [(1, 2), (0, 1)]:
        print("error in prefix_suffix_overlap_hash1")
    if prefix_suffix_overlap_hash2(lst, k) != [(0, 1), (1, 2)] and \
            prefix_suffix_overlap_hash2(lst, k) != [(1, 2), (0, 1)]:
        print("error in prefix_suffix_overlap_hash2")

test()