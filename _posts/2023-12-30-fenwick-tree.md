---
layout: post
title:  "fenwick tree"
date:   2023-12-30 02:00:00 -0800
brief: 'a glimpse into competitive programming'
---

Fenwick tree, also named binary indexed tree (BIT), is a collection of values, used to do effcient computation for either one of the following 3 categories:
- classic:  point update O(logn) & range query  O(logn)
- variant1: point query  O(logn) & range update O(logn)
- variant2: range update O(logn) & range query  O(logn)

### Decription
NOTE: for all the discussion below, we use **1-indexed array**. 

A binary indexed tree, is conceptually represented as a binary tree, and physically represented in an array. We start with the classic binary indexed tree that supports range query and point update. 

For an array `A[1,...n]`, and its binary indexed tree, `B[1,...,n]`, each `B[i]` is a **sum of operations for an incomplete path** that ends at `i`. To find a **sum of operations** of a **complete path** for a given `i`, we use the binary property in numbers represened as binaries. 

To illustrate ragne query, for example, suppose
- the opeartion is mathematical sum
- the goal is to find sum of `A[1], ..., A[7]`, which is the **complete path**

To get the complete path, we take sum of its sub-paths. For `7`, its binary form is `0111`. The number of `1s` is the number of such sub-paths of sums we are looking for, so that the maxium number of sub-paths is constrained to be `log(n)` where `n` is the length of array `A`. 

- for `7`: `B[7] = A[7]`. 1st sub-path. now we remove the rightmost `1` in `7`, `7 = 0111` => `0110 = 6` we have `6` for another exclusive subpath.
- for `6`: `B[6] = A[5] + A[6]`. 2nd sub-path. now we remove the rightmost `1` in `6`, `6 = 0110` => `0100 = 4` we have `4` for another exclusive subpath.
- for `4`: `B[4] = A[4] + A[3] + A[2]`. and `B[2] = A[1] + A[2]`. 3rd subpath. now we remove the rightmost `1` in `4`, `4 = 0100` => `0000 = 0`, we have no subpaths left and thus found the complete path.

`1st sub path + 2nd sub path + 3rd sub path = A[1] + A[2] + ... + A[7]` that is the complete path for the sum we want.

As illustrated, if a single (such as `sum`) operation takes `O(1)` time, then a range query of such operations will take at most `O(logn)` for an array of length `n`. And a single update of an element in the array will also take only `O(logn)` at most because it is the reverse work flow of the range query. Take `3` for another example to illustrate point update:

suppose
- the opeartion is still mathematical sum
- the goal is to update `A[3]`

- update `3 = 0011` that is `B[3]`. add `rightmost 1 = 0001` to `0011` to get an exlusive subpath, `0100 = 4`.
- update `4 = 0100` that is `B[4]`. add `rightmost 1 = 0100` to `0100` to get an exlusive subpath, `1000 = 8`.
- update `8 = 1000` that is `B[8]`. we reach the length of `A` and has completed the point update for all possible subpaths.

Now `3, 4, 8` subpaths will contain `A[3]`. For any `i = [1, 8]`, to find `sum of A[1], ..., A[i]`:
- if `i < 3`: no subpahs in `B[3]` will contain `3`
- if `i = 3`: `B[3]` contains `A[3]`
- if `i = 4`: `B[4]` will jump to `B[3]` that contains `A[3]`
- if `i = 4,5,6,7`, `B[i]` will eventuall hump to `B[4]` that contains `A[3]` because `0100` is a subpath of `0100`, `0101`, `0110`, `0111`
- if `i = 8`, `B[8]` contains `A[3]`

Thus number of operations in point update is also `Olog(n)` as there are at most `log(n)` `1s` for subpaths to update an element. 

#### illustration of a binary indexed tree for an array of length 8
![bit](/assets/images/bit.png)

