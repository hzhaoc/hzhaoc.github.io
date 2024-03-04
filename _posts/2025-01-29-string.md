---
layout: post
title:  "String"
date:   2025-01-29 02:00:00 -0800
brief: 'string pattern matching and so on'
---


# Prefix-Suffix / KMP
`pi[i]` is length of longest suffix ending at `i` that matches prefix starting at `0`

```python
n = len(s)
pi = [0]*n
for i in range(1, n):
    j = pi[i-1]
    while j > 0 and s[i] != s[j]:
        j = pi[j-1]
    if s[i] == s[j]:
        j += 1
    pi[i] = j
```

# Prefix / Z-function
`z[i]` is length of longest prefix starting at `i` that matches the substring starting at `0`

```python
n = len(s)
z = [0]*n
l, r = 0, 0
# note: z[0] is not defined
for i in range(1, n):
    if i < r:
        z[i] = min(r-i, z[i-l])
    while i+z[i]<n and s[i+z[i]] == s[z[i]]:
        z[i] += 1
    if i+z[i] > r:
        l = i
        r = i+z[i]
```