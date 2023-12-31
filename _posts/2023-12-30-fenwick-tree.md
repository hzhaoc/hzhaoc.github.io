---
layout: post
title:  "Fenwick Tree"
date:   2023-12-30 15:25:00 -0800
brief: 'a glimpse into competitive programming'
---

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
    
    def rangeQuery(self, r, l):
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
            k == k & -k
        return
```

#### 2D Form
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
        self.add(self.b2, r+1, v*(l-1) - v*r)

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