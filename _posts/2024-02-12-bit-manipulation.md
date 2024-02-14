---
layout: post
title:  "Bits"
date:   2024-02-12 02:00:00 -0800
brief: 'tricks of bit manipulation'
---


### right most set bit
for `k`, its right most set bit is `k & -k`




### all combinations of subset sum of array
to represent all combinations of subset sum of an array `A`, if `A` has only positive integers, we can represent it with a very large binary number such as `0101111...` where `1`s distance to least significant bit (e.g. `1` in `100` is `2`) is a sum of some subset of `A`.

we can do

```python
sums = 0
for a in A:
    sums |= (sums << a) | (1 << a)
```

- application
leetcode 805. Split Array With Same Average

