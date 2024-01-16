---
layout: post
title:  "Fenwick Tree & Segment Tree"
date:   2024-01-14 19:50:00 -0800
brief: 'a glimpse into competitive programming'
---

*All for range operations*

Fenwick tree, also named binary indexed tree (BIT), is a collection of values, used to do efficient set of operations for either one of the following 3 sets:
- classic:  point update O(logn) & range query  O(logn)
- variant1: point query  O(logn) & range update O(logn)
- variant2: range update O(logn) & range query  O(logn)

## Description
NOTE: for all the discussion below, we use **1-indexed array**.

A binary indexed tree, is conceptually represented as a tree, and physically represented as an array. We start with the classic binary indexed tree that supports range query and point update. 

For an array `A[1,...n]`, and its binary indexed tree, `B[1,...,n]`, each `B[i]` stores **result of a set of accumulated operations for an incomplete path** that ends at `i`. To find a **result of accumulated operations** of a **complete path** for a given `i`, we take advantage of numbers in its binary form.

To illustrate ragne query, for example, suppose
- the opeartion is mathematical sum
- the goal is to find sum of `A[1], ..., A[7]`, which is the **complete path**

To get the complete path, we take sum of its sub-paths. For `7`, its binary form is `0111`. The number of `1s` is the number of such sub-paths of sums we are looking for, so that the maxium number of sub-paths is constrained to be `log(n)` where `n` is the length of array `A`. 

- for `7`: `B[7] = A[7]`. 1st sub-path. now we use `k = k - k & -k` trick to remove the rightmost `1` in `7`, `7 = 0111` => `0110 = 6` we have `6` for another exclusive subpath. 
- for `6`: `B[6] = A[5] + A[6]`. 2nd sub-path. now we remove the rightmost `1` in `6`, `6 = 0110` => `0100 = 4` we have `4` for another exclusive subpath.
- for `4`: `B[4] = A[4] + A[3] + A[2]`. and `B[2] = A[1] + A[2]`. 3rd subpath. now we remove the rightmost `1` in `4`, `4 = 0100` => `0000 = 0`, we have no subpaths left and thus found the complete path.

`1st sub path + 2nd sub path + 3rd sub path = A[1] + A[2] + ... + A[7]` that is the complete path for the sum we want.

As illustrated, if a single (such as `sum`) operation takes `O(1)` time, then a range query of such operations will take at most `O(logn)` for an array of length `n`. And a single update of an element in the array will also take only `O(logn)` at most because it is the reverse work flow of the range query. Take `3` for another example to illustrate point update:

suppose
- the opeartion is still mathematical sum
- the goal is to update `A[3]`
\

we do the following:
- update `3 = 0011` that is `B[3]`. use `k = k + k & -k` trick to add `rightmost 1 = 0001` to `0011` to get an exlusive subpath, `0100 = 4`.
- update `4 = 0100` that is `B[4]`. add `rightmost 1 = 0100` to `0100` to get an exlusive subpath, `1000 = 8`.
- update `8 = 1000` that is `B[8]`. we reach the length of `A` and has completed the point update for all possible subpaths.

Now `3, 4, 8` subpaths will contain `A[3]`. For any `i = [1, 8]`, to find `sum of A[1], ..., A[i]`:
- if `i < 3`: no subpahs in `B[3]` will contain `3`
- if `i = 3`: `B[3]` contains `A[3]`
- if `i = 4`: `B[4]` will jump to `B[3]` that contains `A[3]`
- if `i = 4,5,6,7`, `B[i]` will eventuall hump to `B[4]` that contains `A[3]` because `0100` is a subpath of `0100`, `0101`, `0110`, `0111`
- if `i = 8`, `B[8]` contains `A[3]`

Thus number of operations in point update is also `Olog(n)` as there are at most `log(n)` `1s` for subpaths to update an element. 

### illustration of a binary indexed tree for an array of length 8
![bit](/assets/images/bit.png)


## Point Update & Range Query
Previous description has talked about this type of tree so we directly go for implementation.

```python
class Bit:
    def __init__(self, n):
        # n is length of array
        # 1-based
        self.n = n
        self.b = [0] * (n+1)
    
    def rangeQuery(self, l, r):
        return self.presum(r) - self.presum(l-1)

    def presum(self, k):
        s = 0
        while k > 0:
            s += self.b[k]
            k -= k & -k
        return s

    def pointUpdate(self, k, diff):
        while k <= self.n:
            self.b[k] += diff
            k += k & -k
        return
```

### Point Update & Range Query: Variant 1: 2D Range Query
To do range query and point update in a 2D array. For example, we want to do efficient element update and presum for a matrix `A`, where presum for `A[I][J]` is sum of all `A[i][j]` where `0 <= i <= I and 0 <= j <= J for all possible i, j`.
\
Now in the binary indexed tree, 
- to do such prefix sum, we first need to find all subpaths of rows, then in each row we have to find subpaths of columns. Then for a matrix of length `nxm`, such a presum operation takes `logmlogn`. 
- to do point update, we still reverse the action.

##### code
```python
class Bit:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.b = [[0] * (m+1) for i in range(n+1)]

    def rangeQuery(self, i, j, ni, nj):
        # ni, nj: position of bottom right of the rectangle
        # i, j: position of the top left of the rectangle
        # range sum of the rectangle covered by i,j,ni,nj
        br = self.presum(ni, nj)
        bl = self.presum(ni, j-1)
        tr = self.presum(i-1, nj)
        tl = self.presum(i-1,j-1)
        return br - bl - tr + tl

    def presum(self, i, j):
        s = 0
        while i > 0:
            y = j
            while y > 0:
                s += self.b[i][y]
                y -= y & -y
            i -= i & -i
        return s
    
    def pointUpdate(self, i, j, diff):
        v = diff
        while i <= self.n:
            y = j
            while y <= self.m:
                self.b[i][y] += v
                y += y & -y
            i += i & -i
```

### Point Update & Range Query: Variant 2: range sum for different kinds
It is better to directly look at an example. 
\

> LeetCode 673, Number of Longest Increasing Subsequence: Given an integer array `nums`, return the number of longest increasing subsequences. Notice that the sequence has to be strictly increasing.

\
It is doable in `O(n^2)` with dp: to find count of max length for subarray ending at `i`, we check count and max len for each subarray ending at `j` where `j < i`, and update it for `i`:
- if `nums[j] < nums[i]`: (note `length[i]` initialized to be `0`)
  - if `length[j]+1 == length[i]`: `count[i] += count[j]`
  - if `length[j]+1 > length[i]` : `count[i] = count[j]`, `length[i] = length[j]+1`
\
finally we loop through `length` and `count` arrays and can find sum of count for max length.

Complete code:
```python
# O(n^2)
    def findNumberOfLIS1(self, A: List[int]) -> int:
        n = len(A)
        l = [0]*n
        c = [0]*n
        res = 0
        mal = 0
        for i in range(n):
            l[i] = 1
            c[i] = 1
            for j in range(i):
                if A[j] < A[i]:
                    if l[j]+1 == l[i]:
                        c[i] += c[j]
                    elif l[j]+1 > l[i]:
                        c[i] = c[j]
                        l[i] = l[j]+1
            if mal < l[i]:
                mal = l[i]
                res = c[i]
            elif mal == l[i]:
                res += c[i]
        return res
```

To save time on the inner loop, a bit variant is used here - in the bit, it stores two trees: one for max length for each pos, one for count for each pos. 
```python
    def findNumberOfLIS(self, A: List[int]) -> int:
        # for same value, bigger index should be visited earlier than smaller index, because it is string increasing sequence we want
        A = sorted([[a, i+1] for i, a in enumerate(A)], key = lambda x: [x[0], -x[1]])
        n = len(A)
        b = Bit(n)
        for a, i in A:
            l, c = b.get(i-1)
            if l == 0:
                c = 1
            b.set(i, l+1, c)
        return b.get(n)[1]


class Bit:
    def __init__(self, n):
        self.n = n
        self.l = [0]*(n+1)
        self.c = [0]*(n+1)

    def get(self, i):
        # get sum of cnt c for range (1, i) for max len l
        l = 0
        c = 0
        while i > 0:
            if self.l[i] > l:
                l = self.l[i]
                c = self.c[i]
            elif self.l[i] == l:
                c += self.c[i]
            i -= i & -i
        return l, c

    def set(self, i, l, c):
        # add cnt c for len l at pos i
        while i <= self.n:
            if self.l[i] == l:
                self.c[i] += c
            elif self.l[i] < l:
                self.l[i] = l
                self.c[i] = c
            i += i & -i
```

## Point Query and Range Update
To do this, the reverse of the previous type, given array `A[1,2,..,8]`, to update for range `l,l+1,...,r`, we can use the previous `pointUpdate` binary technique to update `l`, such that all mutually exclusive paths that can contain `l` will contain `l`. We then do reverse-update for `r+1`, such that all mutually exclusive paths after `r` cancels previous update for `l`. 

Take `rangeUpdate(3, 6) to add 1` for example,
- first update `3`: add `1` to `0011`, then `0100`, `1000`. 
- then update `6+1=7`: add `-1` to `0111`, then `1000`. 

now we want to do `pointUpdate(i)`:
- if `i < 3`: we do prefix sum of `3` and see no `1` added.
- if `3 <= i <= 6`: we do prefix sum of `i` and see `1` added, because `0011`, `0100`, `0101`, `0110` will contain subpath of `0100`.
- if `i >= 7`: we do prefix sum of `i` and see no `1` added. because `0111`, `1000` has been reverse-updated and cancelled `1`.

```python
class Bit:
    def __init__(self, n):
        # 1-based
        self.n = n
        self.b = [0] * (n+1)

    def rangeUpdate(l, r, diff):
        self.add(l, diff)
        self.add(r+1, -diff)

    def add(self, k, diff):
        while k <= self.n:
            self.b[k] += diff
            k += k & -k
    
    def pointQuery(self, i):
        return self.presum(i)

    def presum(self, k):
        s = 0
        while k > 0:
            s += self.b[k]
            k -= k & -k
        return s
```

## Range Update and Range Query
Supppose initial array values are `0` and we use previous range update technique to update `v` for range `l...r` for array `A[1...8]`.

now we have an array
- value: `0,0,...,v,...,v,0, ...0  `
- index: `1,2,...,l,...,r,r+1...n-1`

we want to do presum of `i` 
- if `i < l`: `presum = 0`
- if `l <= i <= r`: `presum = v*(i-(l-1))`
- if `i > r`: `presum = v*(r-(l-1))`

to make it clearer: 
- if `i < l`: `presum = 0*i - 0`
- if `l <= i <= r`: `presum = v*i - v*(l-1)`
- if `i > r`: `presum = 0*i - ( v*(l-1) - v*r )`

notice for each expression we break it into two terms. For the first term, we can use a binary indexed tree `b1` to model it: range update of `v` for `l,r` and presum is then just `pointQuery(b1, i)*i`. We use a second binary indexed tree `b2` to model the second term. update `v*(l-1)` for `l,l+1...` and `( v*(l-1) - v*r )` for `r+1,...`. and presume is then just `pointQuery(b2, i)`. Finally combining the two trees we have 
- `presum(i) = pointQuery(b1, i)*i - pointQuery(b2, i)`

then
- `rangeQuery(l, r) = presum(r) - presum(l-1)`

### code
```python
class Bit:
    def __init__(self, n):
        # 1-indexed
        # operation is sum
        self.n = n
        self.b1 = [0] * (n+1)
        self.b2 = [0] * (n+1)

    def rangeQuery(self, l, r):
        return self.presum(r) - self.presum(l-1)

    def rangeUpdate(self, l, r, diff):
        v = diff
        self.add(self.b1, l, v)
        self.add(self.b1, r+1, -v)
        self.add(self.b2, l, v*(l-1))
        self.add(self.b2, r+1, -v*r)

    def add(self, b, k, diff):
        # add diff to k and subpaths above
        while k <= self.n:
            b[k] += diff
            k += k & -k

    def preSum(self, k):
        return self._presum(self.b1, k) * k - self._presum(self.b2, k)

    def _presum(self, b, k):
        s = 0
        while k > 0:
            s += b[k]
            k -= k & -k
        return s
```



# Segment Tree
Segment Tree serves a similar purpose to Fenwick Tree: it provides efficient range operation of some sort. Each node has a value that is a merged of some sort (e.g. sum) from its children nodes. 

### Point Update & Range Query
Here's an implementation. 

```python
class SegTree:
    def __init__(self, n):
        self.root = self._init(0, n-1)

    def _init(self, l, r):
        node = Node(0, l, r)
        if l==r:
            return node
        node.left = self._init(node.l, node.m)
        node.right = self._init(node.m+1, node.r)
        return node
    
    def get(self, l, r):
        return self._get(self.root, l, r)

    def _get(self, node, l, r):
        # (l, r) <= (node.l, node.r)
        if node.l == node.r:
            return node.v
        if node.l == l and node.r == r:  
            # this makes get O(logn)
            return node.v
        if node.m+1 <= l:
            return self._get(node.right, l, r)
        if node.m >= r:
            return self._get(node.left, l, r)
        return self._get(node.left, l, node.m) + self._get(node.right, node.m+1, r)

    def set(self, i, v):
        self._set(self.root, i, v)

    def _set(self, node, i, v):
        if node.l == node.r:
            dv = v - node.v
            node.v = v
            return dv
        dv = 0
        if node.m >= i:
            dv = self._set(node.left, i, v)
        else:
            dv = self._set(node.right, i, v)
        node.v += dv
        return dv


class Node:
    def __init__(self, v, l, r):
        self.v = v
        self.l = l
        self.r = r
        self.m = (l+r)>>1
        self.left = None
        self.right = None
```

Here's an array implementation of the above node approach. Note `set` is a range set. Sometimes if we need to fast non-repeatable range set operation, such as set 0s to 1s for a range (you don't want to reset a range of 1s to 1s.. that's a waste of time), this comes useful.
```python
class Seg:
    def __init__(self, n):
        # 1-indexed
        self.n = 1 << math.ceil(math.log(n, 2)+1)
        self.k = n
        self.b = [0]*self.n

    def add(self, j, v):
        self._add(self, b, 1, 1, self.k, j, v)

    def _add(self, b, i, l, r, j, v):
        # j is in [l, r]
        if l == L and r == R:
            dv = v - b[i]
            b[i] = v
            return dv
        m = (l+r)>>1
        if m >= j:
            dv = self._get(b, i<<1, l, m, j, v)
        else:
            dv = self._get(b, i<<1|1, m+1, r, j, v)
        b[i] += dv
        return dv

    def set(self, l, r):
        # range set 0s to 1s
        # [l, r]
        return self._set(self.b, 1, 1, self.k, l, r)
    
    def _set(self, b, i, l, r, L, R):
        # [L, R] <= [l, r]
        cap = r-l+1

        if b[i] == cap:
            return 0

        if l == r:
            b[i] = 1
            return 1

        m = (l+r)>>1
        ld, rd = 0, 0
        if m >= R:
            ld = self._set(b, i<<1, l, m, L, R)
        elif m+1 <= L:
            rd = self._set(b, i<<1|1, m+1, r, L, R)
        else:
            ld = self._set(b, i<<1, l, m, L, m)
            rd = self._set(b, i<<1|1, m+1, r, m+1, R)
            
        b[i] += ld+rd
        return ld+rd

    def get(self, l, r):
        return self._get(self.b, 1, 1, self.k, l, r)

    def _get(self, b, i, l, r, L, R):
        # [L, R] <= [l, r]
        if l == L and r == R:
            return b[i]

        m = (l+r)>>1
        if m >= R:
            return self._get(b, i<<1, l, m, L, R)
        elif m+1 <= L:
            return self._get(b, i<<1|1, m+1, r, L, R)
        return self._get(b, i<<1, l, m, L, m) + self._get(b, i<<1|1, m+1, r, m+1, R)
        
```

Here's a variant of array implementation. This variant is iterative and bottom-up on array. It works only on point update. 
```python
class SegArr:
    def __init__(self, A):
        self.n = len(A)
        self.t = [0]*n + A
        for i in range(self.n-1, -1, -1):
            self.t[i] += self.t[i<<1] + self.t[t<<1|1]
    
    def get(self, i, j):
        i += self.n
        j += self.n
        res = 0
        while i <= j:
            if i & 1:
                res += self.t[i]
                i += 1
            i >>= 1
            if not (j & 1):
                res += self.t[j]
                j -= 1
            j >>= 1
        return res

    def set(self, i, v):
        i += self.n
        dv = v - self.t[i]
        while i > 0:
            self.t[i] += dv
            i >>= 1
```


### application: Dynamic Segment Tree
An advantage of node-structured segment tree is it does not have to create sub-segments at initialization, different from its list implmeention or binary indexed array-based tree. It is useful when length of a segment tree is too large to completely build upfront and/or when it is beneficial to collapse two sub-segments into a parent segment to save space and time. I took an example from LeetCode which I think is classic to me:

> LeetCode 715. Range Module

> A Range Module is a module that tracks ranges of numbers. Design a data structure to track the ranges represented as half-open intervals and query about them.

> A half-open interval [left, right) denotes all the real numbers x where left <= x < right.

> Implement the RangeModule class:

> RangeModule() Initializes the object of the data structure.
> void addRange(int left, int right) Adds the half-open interval [left, right), tracking every real number in that interval. Adding an interval that partially overlaps with currently tracked numbers should add any numbers in the interval [left, right) that are not already tracked.
> boolean queryRange(int left, int right) Returns true if every real number in the interval [left, right) is currently being tracked, and false otherwise.
> void removeRange(int left, int right) Stops tracking every real number currently being tracked in the half-open interval [left, right).

> Constraints:

> 1 <= left < right <= 10^9
> At most 104 calls will be made to addRange, queryRange, and removeRange.

In this example, if we build a range-query and update based tree completely at initialization, we need to store an array of at least `2*10^9` which is big. So we start with a segment tree with empty intervals, and build only necessary intervals as queries come and go, and collapse sub-intervals into parent when condition is met for optimization. 

#### code
```python
class RangeModule:

    def __init__(self):
        self.b = SL()
        # self.b = Seg(1, 10**9)

    def addRange(self, left: int, right: int) -> None:
        self.b.set(left, right)

    def queryRange(self, left: int, right: int) -> bool:
        return self.b.get(left, right)

    def removeRange(self, left: int, right: int) -> None:
        self.b.unset(left, right)


class Node:
    def __init__(self, l, r, v):
        self.l = l
        self.r = r
        self.set = v  # 1 means all covered
        self.left = None
        self.right = None

class Seg:
    def __init__(self, l, r):
        self.root = Node(l, r, 0)

    def set(self, l, r):
        self._set(self.root, l, r)

    def unset(self, l, r):
        self._unset(self.root, l, r)

    def _set(self, node, l, r):
        # [l, r) is in [node.l, node.r)

        # whole block is set
        if node.set:
            return
        
        # sub-block to set
        if l == node.l and r == node.r:
            node.set = 1
            node.left = None
            node.right = None
            return
        
        # not whole block is set; create sub-blocks for (l, r)
        m = (node.l+node.r)>>1
        if m >= r:
            self._node(node, 1, 0)
            self._set(node.left, l, r)
        elif m <= l:
            self._node(node, 0, 0)
            self._set(node.right, l, r)
        else:
            self._node(node, 1, 0)
            self._set(node.left, l, m)
            self._node(node, 0, 0)
            self._set(node.right, m, r)
        
        # if both children are set - collapse into this root
        if node.left and node.left.set and node.right and node.right.set:
            node.left = None
            node.right = None
            node.set = 1

    def _unset(self, node, l, r):
        # [l, r) is in [node.l, node.r)

        # range covers the block. unset it and return 1 to indicate this block is completely unset (for node collapsing)
        if l == node.l and r == node.r:
            node.set = 0
            node.left = None
            node.right = None
            return 1

        # range is in some sub-blocks
        # - unset and remove fully-unset sub-blocks the range covers
        # - create fully-set sub-blocks the range does not cover
        # - this set block bit has to be 0

        m = (node.l+node.r)>>1
        fullyUnsetLeft = 0
        fullyUnsetRight = 0
        if node.set:
            self._node(node, 1, 1)
            self._node(node, 0, 1)
            if m >= r:
                fullyUnsetLeft = self._unset(node.left, l, r)
            elif m <= l:
                fullyUnsetRight = self._unset(node.right, l, r)
            else:
                fullyUnsetLeft = self._unset(node.left, l, m)
                fullyUnsetRight = self._unset(node.right, m, r)
        else:
            if m >= r:
                fullyUnsetLeft = not node.left or self._unset(node.left, l, r)
            elif m <= l:
                fullyUnsetRight = not node.right or self._unset(node.right, l, r)
            else:
                fullyUnsetLeft = not node.left or self._unset(node.left, l, m)
                fullyUnsetRight = not node.right or self._unset(node.right, m, r)

        node.set = 0

        if fullyUnsetLeft:
            node.left = None
        if fullyUnsetRight:
            node.right = None
        if fullyUnsetLeft and fullyUnsetRight:
            return 1
        return 0

    def get(self, l, r):
        return self._get(self.root, l, r)

    def _get(self, node, l, r):
        # [l, r) is in [node.l, node.r)

        # this block is fully set -> 1
        if node.set:
            return 1

        # this block is fully unset -> 0
        if not node.left and not node.right:
            return 0

        # this block is partially set (it has set children not collapsed), and range covers block -> 0
        if l == node.l and r == node.r:
            return 0
        
        # this block is partially set, and range is in some sub-blocks -> check sub-blocks
        m = (node.l+node.r)>>1
        if m >= r:
            return node.left and self._get(node.left, l, r)
        if m <= l:
            return node.right and self._get(node.right, l, r)
        return node.left and self._get(node.left, l, m) and node.right and self._get(node.right, m, r)

    def _node(self, node, isLeft, setBit):
        m=(node.l+node.r)>>1
        if isLeft and not node.left:
            node.left = Node(node.l,m,setBit)
        if not isLeft and not node.right:
            node.right = Node(m,node.r,setBit)
```

##### bit vs segment tree vs balanced BST, (or SortedList in python)
I've been playing around with bit, seg tree and balanced bst recently, in many cases they are interswtichable because they serve similar purpose: range query and point update or limited range update. I usually go with bit because I saw slightly faster perforamnce with it and its code is more concise relatively speaking. 

segment tree is also handy. it's dynamic tree is sometimes a bit hard to grasp and requires lots of trivial coding in interviews. but it is still very powerful.

"SortedList" , or balanced BST is most intuive and most implementation-complexity hidden because of its complexity. therefore for programming contest, if there's a use case to directly use it i would suggest so. Here's an application of `sorted List` for the above application `range module`

```python
class SL:
    def __init__(self):
        from sortedcontainers import SortedList
        self.sl = SortedList()

    def set(self, l, r):
        sl = self.sl
        i = sl.bisect_left((l,l))
        j = sl.bisect_right((r,r))
        if i and sl[i-1][1] >= l:
            i -= 1
        if j<len(sl) and sl[j][0] <= r:
            j += 1
        if i == j:
            sl.add((l, r))
            # print('after add', sl)
            return
        toRmv = []
        for ll, rr in sl[i:j]:
            l = min(l, ll)
            r = max(r, rr)
            toRmv.append((ll, rr))
        for ll, rr in toRmv:
            sl.remove((ll, rr))
        sl.add((l, r))
        # print('after add', sl)

    def unset(self, l, r):
        sl = self.sl
        i = sl.bisect_left((l,l))
        j = sl.bisect_right((r,r))
        if i and sl[i-1][1] >= l:
            i -= 1
        if j<len(sl) and sl[j][0] <= r:
            j += 1
        if i == j:
            return
        toRmv = []
        toAdd = []
        if sl[j-1][1] > r:
            toAdd.append((r, sl[j-1][1]))
        if sl[i][0] < l:
            toAdd.append((sl[i][0], l))
        for ll, rr in sl[i:j]:
            toRmv.append((ll, rr))
        for ll, rr in toRmv:
            sl.remove((ll, rr))
        for ll, rr in toAdd:
            sl.add((ll, rr))
        # print('after remove', sl)

    def get(self, l, r):
        sl = self.sl
        i = sl.bisect_left((l,l))
        if i and sl[i-1][1] > l:
            i -= 1
        j = sl.bisect_right((r,r))
        if j<len(sl) and sl[j][0] < r:
            j += 1
        if i == j-1 and sl[i][0] <= l and sl[i][1] >= r:
            # print(f'get true, {l, r} on {sl}')
            return True
        # print(f'get false, {l, r} on {sl}')
        return False
```
