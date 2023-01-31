+++
title = "Optimization - Caching"
date = 2023-01-27
+++

In this post I take notes on how to optimize python code and also try to define
possible approaches in this process. This will be a series of posts, where later
I expect to cover profiling, where we can have a deep diagnosis on how to improve
our code.

I will take long time discussing algorithmic complexity so I will just jump to an
exponential complexity code, and then present many ways on how to optimize it. So
here is our exponential complexity program.

```python
#!/usr/bin/python3

def fib(a):
    if a <= 1:
        return 1
    return fib(a-1) + fib(a-2)

for i in [10, 20, 50]:
    print(fib(i))
```

This code call the function fib many times, recomputing the same results. A simple
solution is use a dict to keep previous computed data stored.

```python
#!/usr/bin/python3

num = {0: 1, 1:1}

def fib(a):
    if a in num:
        return num[a]
    return fib(a-1) + fib(a-2)

for i in [10, 20, 50]:
    print(fib(i))
```
Also with functools you can cache the results with the cache decorator. So it will
store the function calls.

```python
#!/usr/bin/python3
from functools import cache

@cache
def fib(a):
    if a <= 1:
        return 1
    return fib(a-1) + fib(a-2)

for i in [10, 20, 50]:
    print(fib(i))
```

The only restriction is that the parameters of the function must be hashable and
it has no side effects. Another possible approch is instead of caching to memory,
caching to disk, so the computed data could be reused between different runs.

```python
#!/usr/bin/python3
from joblib import Memory

location = './cachedir'
memory = Memory(location, verbose=0)

@memory.cache
def fib(a):
    if a <= 1:
        return 1
    return fib(a-1) + fib(a-2)

for i in [10, 20, 50]:
    print(fib(i))
```
