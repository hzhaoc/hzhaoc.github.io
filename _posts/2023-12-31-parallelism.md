---
layout: post
title:  "Parallelism"
date:   2023-12-31 00:10:00 -0800
brief: 'when linear algebra meets distributed computing'
---

# Start from matrix multiplication
We know in a single thread, multiplication of two matrixes in naive implementation can be easily done in `O(n^3)` time where `n` is the side length. There are algorithms that can asymptotically approach the theoraticall trivial lower bound of `O(wn^2)` where `2 <= w < 2.37...`. One of such algorithm is strassen algorithm with a run time of $O(n^{\log_2^{7}})\approx{O(n^{2.8})}$

## matrix multiplication in single thread
```python
class BaseMatrix:
	"""
	implement a matrix class that supports metrix relevant operations
	"""

	def inv(self, A):
		"""
		find inverse matrix of A
		@input: input square matrix A
		@output: inverse of A
		"""
		n = len(A)
		A, P = self._LUPDecompose(A, n)
		IA = [[0.0] * n for i in range(n)]
		# forward/backward substitution to solve IA from L/U and P
		for j in range(n):
			for i in range(n):
				IA[i][j] = 1.0 if P[i] == j else 0.0

				for k in range(i):
					IA[i][j] -= A[i][k] * IA[k][j]

			for i in range(n-1, -1, -1):
				for k in range(i+1, n):
					IA[i][j] -= A[i][k] * IA[k][j]

				IA[i][j] /= A[i][i]

		return IA

	def det(self, A):
		"""
		find determinant of square matrix A
		@input: input square matrix of size n
		@output: determinant of A
		"""
		n = len(A)
		A, P = self._LUPDecompose(A, n)
		det = A[0][0]
		
		for i in range(1, n):
			det *= A[i][i]
		
		return det if (P[n] - n)%2 == 0 else -det

	def mult(self, A, B): 
		""" 
		matrix multiplication with divide and conquer approach (Strassen Algorithm), recursive
		"""
		h, w = len(A), len(B[0])
		A, B = self._strassen_padding(A), self._strassen_padding(B)
		C_padded = self._strassen(A, B)
		C = [C_padded[i][:w] for i in range(h)]
		return C

	def add(self, A, B):
		"""matrix element-wise add"""
		if len(A) != len(B) or len(A[0]) != len(B[0]):
			raise ValueError("two matrixes should have same shape to perform element-wise oeprations")
		return [[A[i][j]+B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

	def sub(self, A, B):
		"""matrix element-wise subtract"""
		if len(A) != len(B) or len(A[0]) != len(B[0]):
			raise ValueError("two matrixes should have same shape to perform element-wise oeprations")
		return [[A[i][j]-B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

	def prod(self, A, B):
		"""matrix element-wise product"""
		if len(A) != len(B) or len(A[0]) != len(B[0]):
			raise ValueError("two matrixes should have same shape to perform element-wise oeprations")
		return [[A[i][j]*B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

	def show(self, A):
		for i in A:
			print(i)

	def _strassen_base(self, A, B):
		"""
		base case for strassen algorithm, default is product of two numbers, 
		since its data structure is nested list, the product is represented by element-wise matrix product
		"""
		return self.prod(A, B)

	def _strassen_padding(self, A):
		w, h = len(A[0]), len(A)
		k = math.log(max(w, h), 2)
		if not k.is_integer():
			n = int(2**(k//1 + 1))
		else:
			n = int(2**k)
		return [A[i] + [0]*(n - w) if i < h else [0]*n for i in range(n)]

	def _strassen(self, A, B): 
	    """ 
	    matrix multiplication by divide and conquer approach, recursive
	    """
	  
	    # Base case when size of matrices is 1x1 
	    if len(A) == 1:
	    	return self._strassen_base(A, B)

	    # Split the matrices into blocks until base case is reached
	    a, b, c, d = self._split(A) 
	    e, f, g, h = self._split(B) 
	  
	    # Computing the 7 matrix multiplications recursively
	    m1 = self._strassen(a, self.sub(f, h))
	    m2 = self._strassen(self.add(a, b), h)         
	    m3 = self._strassen(self.add(c, d), e)         
	    m4 = self._strassen(d, self.sub(g, e))
	    m5 = self._strassen(self.add(a, d), self.add(e, h))
	    m6 = self._strassen(self.sub(b, d), self.add(g, h))   
	    m7 = self._strassen(self.sub(a, c), self.add(e, f))
	  
	    # Computing the values of the 4 blocks of the final matrix c
	    c11 = self.add(self.sub(self.add(m5, m4), m2), m6)
	    c12 = self.add(m1, m2)            
	    c21 = self.add(m3, m4)          
	    c22 = self.sub(self.sub(self.add(m1, m5), m3), m7)
	  
	    # Combining the 4 blocks into a single matrix by stacking horizontally and vertically
	    C = self._vstack(self._hstack(c11, c12), self._hstack(c21, c22))

	    return C

	def _split(self, mat): 
	    """ 
	    Splits a given matrix into 4 blocks 
	    """
	    h, w = len(mat), len(mat[0])
	    h2, w2 = h//2, w//2
	    return [mat[r][:w2] for r in range(h2)], \
	    	   [mat[r][w2:] for r in range(h2)], \
	    	   [mat[r][:w2] for r in range(h2, h)], \
	    	   [mat[r][w2:] for r in range(h2, h)]

	def _hstack(self, M1, M2):
		"""satck two matrixes into one horizontally"""
		w1, w2 = len(M1[0]), len(M2[0])
		h1, h2 = len(M1), len(M2)
		if h1 != h2:
			raise ValueError("two matrixes should have same height to be stacked horizontally")
		M0 = [[0 for x in range(w1+w2)] for y in range(h1)]
		i = 0
		while i < h1:
			M0[i] = M1[i] + M2[i]
			i+=1
		return M0

	def _vstack(self, M1, M2):
		"""satck two matrixes into one vertically"""
		w1, w2 = len(M1[0]), len(M2[0])
		h1, h2 = len(M1), len(M2)
		if w1 != w2:
			raise ValueError("two matrixes should have same width to be stacked vertically")
		M0 = [[0 for x in range(w1)] for y in range(h1+h2)]
		for i in range(h1+h2):
			if i < h1:
				M0[i] = M1[i]
			else:
				M0[i] = M2[i-h1]
		return M0

	def _LUPDecompose(self, A, n):
		"""
		LUP decomposition of matrix A
		@input
		A: input square matrix of dimension n
		@output
		A: matrix A is modified and contains a copy of both matrices L-E and U as A=(L-E)+U such that 
			it is a LUP decomposition where P*A=L*U. The permutation matrix is a integer list of size n+1
			containing column indexes of the represented permutation matrix P where the P has "1". The last 
			element P[n]=s+n, where s is the number of row exchanges of P from idendity matrix E, for the purpose 
			of computing determinant as det(P)=(-1)^S
		P: 1-d list of length n+1 representing permutation matrix and swap count
		"""
		
		if n!= len(A[0]):
			raise ValueError("input is not a square matrix")

		P = [i for i in range(n+1)]  # initiate P as an identity matrix with a permutation counter with initial value = n

		for i in range(n):
			maxA = 0.0
			imax = i

			for k in range(i, n):  # determine index of max value per column in A
				absA = abs(A[k][i])
				if absA > maxA:
					maxA = absA
					imax = k

			if imax != i:  # if max is not in diagonal, swap
				# swap P
				j = P[i]
				P[i] = P[imax]
				P[imax] = j
				# swap rows of A
				row = A[i]
				A[i] = A[imax]
				A[imax] = row
				# count swaps
				P[n]+=1

			for j in range(i+1, n):  # forward substitution for A=(L-E)+U so that P*A=L*U
				A[j][i] /= A[i][i]

				for k in range(i+1, n):
					A[j][k] -= A[j][i] * A[i][k]

		return A, P
```

## matrix multiplication in distributed network

### Preword: Distributed Memory Netowrk Topology 
some definitions or properties of a network...
- **links**: number of node connections
- **diameter**: longest shortest path
	- the lower the better, the less delay
- **bisection width**: minimum number of edge/link cuts to cut network/graph in half
	- the higher, the better bandwidth, the less congestion

> improve diameter: 
> - linear -> ring
> - mesh -> torus
> -  ...

some types of network... (assume p nodes)
- linear
- ring
- mesh
- torus
- tree
- d-dimensional mesh or torus
	-	**many supercomputers (2021) use low dimensional meshes**
-	hyper cubes: logp dimensional torus
	-	high cost of wire, but high bisection width in return

Network Type | Links | Diameter | Bisection
------------ | ------------ | ------------ | ----------
Linear | p-1 | p-1 | 1
2-D Mesh | 2p | $2\sqrt{p}$ | $\sqrt{p}$
Fully Connected | $p(p-1)/2$ | 1 | $p^2/4$
Binary Tree | P | log(p) | 1
D-Dimensional Mesh | dP | $\frac{1}{2}dP^{1/d}$ | $2P^{(d-1)/d}$
Hypercube | plogp | logp | p/2

##### congestion concept
to  map a logical topological network to a physical one...
consider congestion:
- **congestion = maximum number of logical edges that map to a physical edge**

**lower bound of a congestion C = logical bisection / physical bisection**. If you know the congestion. you’ll know how much worse the cost of your algorithm will be on a physical network with a lower bisection capacity.

### Loomis and Whitney
In matrix multiply A = BC. Given any sub-block of sA, sB, sC, 
- Volume of $|I| <= √|sA| |sB| |sC|$

### On distributed network
#### 1D network algorithm
Use Block Row Distribution: each node gets n/P rows. In AB=C, A, or B, say B, has to do row shift.
![1dmatmul](/assets/images/distri%20mem%20mat%20mul%20row%20distri%20algo.png)

Rearrange code to overlap communication latency: (improve run time by mostly 2)

```c++
let A’[1: n/P] [1:n] = local part of A 
	B’, C’ = same for B, C 

let B’’[1:n/P][1:n] = temp storage 
let rnext ← (RANK + 1) mod P 
	rprev ← (RANK + P ­1) mod P 

for L ← 0 to P-1 do 
	sendAsync (B’ → rnext ) 
	recvAsync (B’’ ← rprev ) 
	C’[:][:] += A’[:][...L…] . B’[...L…][:] 
	wait(*) 
	swap(B’, B’’)
```
- Cost: 
	- $T_{comp} (n,P) = 2τ n^3 / P$
		- τ = time per "flop" (flop means …1 floating point multiply or add)
	- $T_{commu} = αP + βn^2$
		- B’ is the only data communicated. It’s size is n/P words by n columns, so n /P words
		- There are P sends that have to be paid for.

- Efficiency
	- Speedup: $S (n; p) = T_*(n) / T_{1D} (n; p) = P / max(1, 1/2 * α/τ * P^2 /n^3 + 1/2 * β/τ * P/n) = θ(P)$
	- Parallel Efficiency = Speedup/P
		- it is good when it's constant > 0. i.e. when $n = \Omega (P)$
		- if you double nodes, you have to quadruple size of the matrices. Number of floating point computations will increase by factor of 8. If you don't quadruple problem size, you will see diminishing speedup efficiency.

- **Isoefficiency**
	- the value of P that n must satisfy to remain constant parallel efficiency. in the example it is $n = \Omega (P)$
	- **measures the scalability of parallel algorithms and architectures**

#### 2D network algorithm
each node in mesh responsible for its corresponding block multiplication. trip by strip, each trip needs to be broadcast
![2d network mat mul algo](/assets/images/2d%20network%20mat%20mul%20algo.png)

- communication time
	- mesh is more scalable than linear network for matrix multiply
		- The bucket is slightly worse than the tree, it trades a higher latency cost for a lower communication cost.

#### beyond 2D - lower communication bound
$T (n; p) = a * √P + b * n^2 √P)$











# More On Parallelism of Distributed Memory

## Message Passing Interface based parallelism(MPI)
MPI defines a standard library for message-passing that can be used to develop portable message-passing programs using either C or Fortran. The MPI standard defines both the syntax as well as the semantics of a core set of library routines that are very useful in writing message-passing programs.
### principals of message passing
The logical view of a machine supporting the message-passing paradigm consists of p processes, each with its own exclusive address space. Instances of such a view come naturally from clustered workstations and non-shared address space multicomputers. There are two immediate implications of a partitioned address space. First, each data element must belong to one of the partitions of the space; hence, data must be explicitly partitioned and placed. This adds complexity to programming, but encourages locality of access that is critical for achieving high performance on non-UMA architecture, since a processor can access its local data much faster than non-local data on such architectures. The second implication is that all interactions (read-only or read/write) require cooperation of two processes – the process that has the data and the process that wants to access the data. This requirement for cooperation adds a great deal of complexity for a number of reasons. The process that has the data must participate in the interaction even if it has no logical connection to the events at the requesting process. In certain circumstances, this requirement leads to unnatural programs. In particular, for dynamic and/or unstructured interactions the complexity of the code written for this type of paradigm can be very high for this reason. However, a primary advantage of explicit two-way interactions is that **the programmer is fully aware of all the costs of non-local interactions**, and **is more likely to think about algorithms (and mappings) that minimize interactions**. Another major advantage of this type of programming paradigm is that it can be efficiently implemented on a wide variety of architectures.

The message-passing programming paradigm requires that the parallelism is coded explicitly by the programmer. That is, the programmer is responsible for analyzing the underlying serial algorithm/application and identifying ways by which he or she can decompose the computations and extract concurrency. As a result, programming using the message-passing paradigm tends to be hard and intellectually demanding. However, on the other hand, properly written message-passing programs can often achieve very high performance and scale to a very large number of processes.


### A Basic Model of Distributed Memory
Assumptions
- network nodes are fully connected, bidirectionally (so communication between two nodes can happen two way simultaneously)
- two nodes communicate messages one at a time
- cost of communication between two nodes are independent of paths taken in the network graph (Later in lesson this will be discussed)
	- time for communicating **n** words
	- $T=\alpha +  \beta n$
- when there's **k** messages competing for same message channel between two nodes, each with **n** words. this is sequential
	- $T=\alpha + \beta n k$

##### Pipelined message delivery
If two nodes are P nodes away in network, two directly connected nodes has message delivery time of t, then transferring n words between these two nodes needs time
$$\alpha + t(P-2) + tn$$
- first term is software preparation overhead
- second is delay
- third is message size 
- first two terms can be thought of constant

##### Point to Point primitives
- assume SPMD style
- handle ← sendAsync(buf[1:n] → dest)
	- the return does not tell you very much. This is done to allow the programmer theability to decide what to do about a send message: wait for ‘message received’ or make acopy and continue to work.
- handle ← recvAsync(buf[1:n] ← source)
	- return means the message was delivered.
- wait(handle1, handle2, ...) or wait(*)

> Remember every send must have a matching receive
> sendAsync and recvAsync can get trapped in a deadlock depending upon how the wait isimplemented.

##### Collectives Operation runtime
- a quick recap: with tree-based algorithm
	- start cost / $\alpha$ term is optimal
	- bandwidth / $\beta$ term is sub-optimal (except for scatter, gather)
	- ![network communication collectives tree based algo run time recap](/assets/images/network%20communication%20collectives%20tree%20based%20algo%20run%20time%20recap.png)
- a quick recap: with additional bucket-based algorithm
	- start cost / $\alpha$ term is sub-optimal (except for scatter, gather)
	- bandwidth / $\beta$ term is optimal by factor of 2
	- ![network communication collectives bucket based algo run time](/assets/images/network%20communication%20collectives%20bucket%20based%20algo%20run%20time%20recap.png)

A collective is an operation that must be executed by all processes.
- **Reduce**: 
	- tree based
		- **T(n) = αlogP + βnlog P** using tree based / divide and conquer.
		- ![network reduce](/assets/images/network%20reduce.png)
		- ```c++
			let s = local value
			bitmask ← 1
			while bitmask < P do
				PARTNER ← RANK xor(^) bitmask
				if RANK & bitmask then
					sendAsync (s[:] → PARTNER)
					wait(*)
					break //one sent, the process can drop out
				else if (PARTNER < P)
					recvAsync(t[:] ← PARTNER)
					wait(*)
					S[:] ← S[:] + t[:]
				bitmask ← (bitmask << 1)
			if RANK = 0 then print(s)
			// RANK(Receiver) < RANK(Sender) < P
	- bucketing reduce scatter -> gather
		- **T(n) = αP + βn**
		
```c++
reduce(A_local[1:n], root)
```
- **Broadcast**: 
	- reverse reduce: **T(n) = αlogP + βnlog P**
	- scatter + all-gather (bucketing): **T(n) = αP + βn**
		- ![network broadcast as scatter + allgather](/assets/images/network%20broadcast%20as%20scatter%20%2B%20allgather.png)
```c++
broadcast(A_local [1:n], root)
```
- **Gather**
	- one node collects all pieces, each from a different node
	- reverse scatter, $T(n)=αlogP+βn$
```c++
gather(In[1:m], Out[1:m][1:P], root)
```
- **Scatter**: reverse of gather
	- naive scatter:  $T(n)=αP+βn$
		- ![distri mem naive scatter algo](/assets/images/distri%20mem%20naive%20scatter%20algo.png)
	- If we divide and conquer this scatter, new time is $T(n) = αlogP + βn ((P − 1 ) / P )=αlogP+βn$
		- ![scatter divide and conquer](/assets/images/scatter%20divide%20and%20conquer.png)
```c++
scatter(In[1:m][1:P], root, Out[1:m])
```
- **All-gather**: 
	- **gather -> broadcast**
		- T(n = mP) = O(a*logp + βn*logp)
	- **bucketing**: when n is large, have each process send its piece to its neighbor in each iteration and go all the way down. this can be down in parallel.
		- T(n = mP) = (α + βn/P)(P − 1) ≈ αP + βn
		- αP is sub-optimal, βn is optimal
```c++
allGather(In[1:m], Out[1:m][1:P])
```
- **Reduce-scatter**
	- reduce and then scatter
	- reverse of all-gather; T = αP + βn
```c++
reduceScatter(In[1:m][1:P], Out[1:m])
```
- **All-reduce**
	- reduce-scatter -> all-gather. 
	- T(n = mP) ≈ αP + βn


## shared-memory based parallelism (OpenMP)
Explicit parallel programming requires specification of parallel tasks along with their interactions. These interactions may be in the form of synchronization between concurrent tasks or communication of intermediate results. In shared address space architectures, communication is implicitly specified since some (or all) of the memory is accessible to all the processors. Consequently,** programming paradigms for shared address space machines focus on constructs for expressing concurrency and synchronization along with techniques for minimizing associated overheads**. In this chapter, we discuss shared-address-space programming paradigms along with their performance issues and related extensions to directive-based paradigms.

Shared address space programming paradigms can vary on mechanisms for data sharing, concurrency models, and support for synchronization. Process based models assume that all data associated with a process is private, by default, unless otherwise specified (using UNIX system calls such as shmget and shmat).

### OpenMP: a Standard for Directive Based Parallel Programming
> ...APIs such as Pthreads are considered to be low-level primitives. Conventional wisdom indicates that a large class of applications can be efficiently supported by higher level constructs (or directives) **which rid the programmer of the mechanics of manipulating threads**. Such directive-based languages have existed for a long time, but only recently have standardization efforts succeeded in the form of OpenMP. OpenMP is an API that can be used with FORTRAN, C, and C++ for programming shared address space machines. OpenMP directives provide support for concurrency, synchronization, and data handling while obviating the need for explicitly setting up mutexes, condition variables, data scope, and initialization.

# considerations for Design Parallel Algorithms
### Automatic
compiler finds any possible parallelization points

### Manual
programmer finds any possible parallelization points
1. Determine if the problem is one that can be solved in parallel.
2. Identify Program Hotspots
3. Identify Bottlenecks
4. Identify Inhibitors to Parallelism
5. If possible look at other algorithms to see if they can be parallelized.
6. Use parallel programs and libraries from third party vendors.

##### partitioning
Divide the tasks to be done into discrete chunks.There are two methods for partitioning:
1. Domain Decomposition
The data is decomposed. Then each task works on a portion of the data.

2. Functional Decomposition
The problem is broken into smaller tasks. Then each task does part of the work that needs to be done. In functional decomposition the emphasis is on the computation, rather than the data.

Examples of Functional Decomposition
1. Ecosystem modeling
2. Signal processing
3. Climate modeling

##### communications
Which tasks need to communicate with each other. If communication is needed between processes
- message passing (explicit) or shared address spacing (implicit)
- synchronous or asynchronous
- scope
	- point to point
	- collective (data sharing between two or more tasks)

##### synchronization
There are 3 kinds of Synchronization:
- barrier
- lock/semaphore
- synchronous communication operations e.g. send & recv

##### load balancing
keep all tasks computing volume even, so no bottleneck.








# Two-Level Memory Model (slow-fast memory model)
key take-away:  caches managed by hardware itself are fast, but not sufficient, therefore we need algorithms for more efficient slow-fast memory transfers.

### notation
- **Q**: number of slow-fast memory transfers
- **L**: block transfer size
- **Z**: fast memory size
- **n**: number of elements in slow memory to operate on

## A First Basic Model
To find a locality aware algorithm we need a machine model - will be using a variation on the von Neumann model.  

von Neumann Model:
- Has a sequential processor that does basic compute operations
- Processor connects to a main memory - nearly infinite but really slow
- Fast memory - small but very fast, size = Z ... measured in number of words

Rules:
1. The processor can only work with data that is in the fast memory, known as the local data rule.
2. When there is a transfer of data between the fast and slow memory, the data is transferred in blocks of size **L**, known as the block transfer rule. In this model you may need to consider data alignment 

Costs: The model has two costs associated with it:
1. Work. W(n) == the # of computation operations. (How many operations will the processor have to perform?)
2. **Data transfers, Q(n;Z;L) == # of L-sized slow-fast transfers **(loads and stores). The number of transfers is dependent upon the size of the cache and block size. This will referred to as **Q** and be called the I/O Complexity. 

## Algorithm Design goals
- **Work optimality**.  two-level memory model should do asymptotic work as the best serial/sequential RAM model
$$w(n)=\Theta{(w_*(n))}$$
- **High Computational Intensity**.  (do NOT sacrifice work optimality for this!). This metric measures the inherent locality of an algorithm.
$$I(n;Z;L)=\frac{operations}{words}=\frac{w(n)}{L*Q(n;Z;L)}$$

### performance
suppose 
- $\tau=\frac{time}{operation}$ (time per unit of computation operation)
- $\alpha=\frac{time}{word}$ (amortized time per word in slow-fast memory transfer)

then 
- $T_{comp}=\tau w$
- $T_{mem}=\alpha LQ$
- $\max (T_{comp},\  T_{mem})_{\text{perfect overlap}}\leq T_{exe} \leq (T_{comp}+T_{mem})_{\text{no overlap}}$

with refactoring
- $T_{exe}\geq \tau w(1+\frac{\alpha / \tau}{W/LQ})$
	- $\alpha / \tau$ is hardware dependent, so-called "**machine balance**". Denote it $B$. machine balance usually increases in trend, due to more improvement on microprocessor than on memory (multi-core, etc.)
	- $W/LQ$ is computational intensity described above (algorithm dependent). Denote it $I$. 

then
- $T_{exe}\geq \tau w(1+\frac{B}{I})$

if we normalize this performance
- $R\leq \frac{\tau w_*}{\tau w(1+\frac{B}{I})}=\frac{w_*}{w}\min (1, I/B)$ where $w_*$ is best serial RAM model's operations. 

we can see that $R$ is capped when $I=B$ which means 
- when $I<B$, algorithm is memory-bound
- when $I>B$, algorithm is compute-bound

##### more..
The intuition from *energy & time efficiency analysis and actual testing*[^1] is, as intensity increases, 
- time efficiency reaches optimal faster than energy efficiency
- when time efficiency just reaches its optimal at $I=B_{time}$, energy efficiency is at roughly half.
- optimal energy efficiency implies optimal time efficiency; optimal time efficiency does not
- balance gap is because $B_{time}>B_{energy}$, due to "constant power and other microarchitectural inefficiencies".


## IO Avoiding Algorithm
Continue analysis for several common IO algorithms:
##### Q (transfers) for merge sort
$$Q(n;Z;L) = \Omega(\frac{(n/L)\log(n/L)}{\log(Z/L)})$$ 
> analysis:
> - Phase 1: for each chunk of fast memory of total $n/Z$ chunks, do sort, takes $ZlogZ$ comparisons, and $Z/L$ transfers. 
>   - total transfer: $n/L$. 
>   - total comparison: $nlogZ$
> - Phase 2: two-way merging. for each sorted chunk of size $Z$ in slow memory, read them into fast mem, and merge two chunks into a bigger sorted chunk, then output. this has $log(n/Z)$ steps. At each step $k$ there's $(n/z)*(1/2)^k$ number of $z^k$ size chunks, 
>   - total transfers: $(n/z)*(1/2)^k*(z^{(k+1)}/L)*log(n/Z) = 2\frac{n}{L}\log\frac{n}{Z}=O(\frac{n}{L}\log\frac{n}{Z})$ 
>      - (in each merge, you need read two chunks from slow to fast and write back same size.). 
>   - total comparisons: $n\log\frac{n}{Z}=O(n\log\frac{n}{Z})$
>   
> Summing up:
>  - transfer: $O(\frac{n}{L}\log\frac{n}{Z})$ a bit higher than lower bound
>  - comparison: $O(nlogZ)$
> 
> If we do K-way merging that fully utilizes fast mem size $Z$, then we can reach theoretical lower bound:
> - transfers: $\Omega(\frac{n}{L}\log_{\frac{Z}{L}}{\frac{n}{L}})$
> - comparison: $O(nlogn)$

##### Q for binary search
$$Q(n;Z;L) = \Omega{\frac{\log n}{\log L}}$$
> according to information theory, an array with $n$ binary bits contains $\log n$ information. For each $L$ transfer, you can learn $\log L$ information.
> 
> van emeda boas data layout in fast memory achieves this lower bound: (DATA STRUCTURE MATTERS.)
> - recursively partition a binary search tree by half (equal height) into subtrees, align divided subtrees linearly in consecutive memory space (each base subtree must fit into a cache line of course)
> (this is a **cache-oblivious**)
> - ![binary search lower io bound from van emeda boas](/assets/images/binary%20search%20lower%20io%20bound%20from%20van%20emeda%20boas.png)

##### Q for matrix multiplication
$$Q(n;Z;L) = \Omega{(n^3/(L\sqrt{Z}))}$$ 
- proof 1: (this is a **cache-aware**)
	- ![matrix multiply block transfer](/assets/images/matrix%20multiply%20block%20transfer.png)
- proof 2 from a different divide and conquer algorithm: (this is a **cache-oblivious**)
	- operations: $O(2n^3)$
	- transfers: $Q(n;Z;L) = \Omega{(n^3/(L\sqrt{Z}))}$
	- ![matrix multiply divide and conquer](/assets/images/matrix%20multiply%20divide%20and%20conquer.png)

### more about Cache oblivious..
the term Cache oblivious refers to an IO algorithm whose $Q$ is irrelevant to cache size $Z$. for example, sum of $n$ size array takes $Q=O(n/L)$. for a counter example, matrix multiplication in block transfer is cache ware.

##### LRU-OPT Competitiveness Lemma
the lemma[^2] says $$Q_{LRU}(n;Z;L)\leq 2Q_{OPT}(n;\frac{n}{2};L)$$ It means that number of transfers for a LRU-based cache can be asymptotically close to number of transfers for an cache with optimal replacement policy but only half of size. 

##### Corollary (Regularity Condition)
$Q_{OPT}(n;Z;L)=O(Q_{OPT}(n;2*Z;L))$. For example, previously we see for matrix multiplication, $Q(n;Z;L) = \Omega{(n^3/(L\sqrt{Z}))}$, when $Z$ doubles, there's only constant factor change of optimal number of transfer.

Once the $Q$ obeys this condition, then the lemma stays.

##### tall-cache assumption
an interesting design assumption of cache where its height (number of cache lines) shall be larger than its width (number of words in a line). this will be helpful when algorithm is related to matrix block transfer. for example, a $b*b$ block transfer requires $Z\geq b*b$

most actual caches, probably except for TLB, are indeed tall.


[^1]: Jee Whan Choi, et al, A roofline model of energy, 2012.05
[^2]: Frigo et al, (FOCS, 1999)





# Work Span Model
Work-Span model is implemented by a DAG (directed acyclic graph) where each node as a work depends on one another. 

Assume:
1. All processors run at same speed
2. 1 operation work = 1 unit of time
3. no edge cost

Denote:
- total work: $W(n)$
- depth of DAG: $D(n)$
- number of processors: $p$
- total execution time: $T_p{n}$

### Analysis
- $T_p{n}\geq\max \{D(n), \ \ \text{Ceil}(\frac{W(n)}{p})\}$
	- Span Law: $T_p{n}\geq D(n)$
	- Work Law: $T_p{n}\geq\text{Ceil}(\frac{W(n)}{p})$

##### Brent's Theorem
break execution in phases:
- each phase has 1 vertex in critical path (longest path in DAG)
- non-critical path vertex in each phase are independent
- every vertex must appear in some phase

extra denote:
- execution time in $k$th phase: $t_k$
- work in $k$th phase: $w_k$

then:  
$$t_k=\text{Ceil}(\frac{w_k}{p})=>T_p=\sum_{k=1}^D{\text{Ceil}(\frac{w_k}{p})}\leq\sum_{k=1}^D\text{Floor}(\frac{w_k-1}{p}+1)=\frac{W-D}{p}+D$$

to sum up: $$\max \{D(n), \ \text{Ceil}(\frac{W(n)}{p})\}\leq T_p\leq{\frac{W-D}{p}+D}$$

##### Speedup
Consider speedup on our DAG parallel algorithm calculated as $$\frac{best\ sequential \ time}{parallel\ time}$$
namely $$S_p(n)=\frac{T_*(n)}{T_p(n)}$$
substitution by previous inequality: $$S_p(n)\leq \frac{p}{\frac{W}{W_*}+\frac{p-1}{W_*/D}}$$

you can clearly see if speedup wants to scale linearly with $p$, this two equation must be met in the scaling inequality:
- work optimality: $$W(n)=O(W_*(n))$$
- weak scalability: $$p=O(\frac{W_*}{D})$$

### basic concurrency primitives
![basic concur primitives.png|600](/assets/images/basic%20concur%20primitives.png)
-  spawn: child thread
-  sync
-  par-for (parallel loop, generate independent child for each iteration)
	-  **if iteration operation is independent of each other, the total span will be $O(logn)$ by divide and conquer, not theoretical $O(1)$ since all iterations can not be spawned all at once.**  See below the assumed implementation for par-for in the course CSE 6220.
	-  ![par-for implementation.png|550](/assets/images/par-for%20implementation.png)

### desiderata for work-span
- work optimality: $$W(n)=O(W_*(n))$$
- "low" span (poly-logarithmic): $$D(n)=O(log^kn)$$

motivation: $\frac{W}{D}=O(\frac{n}{log^kn})$ grows nearly linearly with $n$

##### matrix-vector multiply
- loop1 operation is independent
- loop2 operation is also independent with a temp value
![work-span matrix mult.png|600](/assets/images/work-span%20matrix%20mult.png)

![par bfs.png|500](/assets/images/par%20bfs.png)

Depth of BFS search should be "diameter" of graph, or the longest distance between two nodes (i.e. the number of waves from one node to spread over to another node)

### Algorithm
pseudo code
-  l ← 0 ..... the code is level synchronous. This is the level counter set to 0
-  The frontiers referenced are also level specific (Fl)
-  l ← l + 1 .... increments the counter
-  The span (defined by the while loop) will be no larger than the diameter of the graph
-  Process level will: take the graph and the current frontier. It will produce a new frontier and update the distances (D)
![par bfs pseudo  code.png|400](/assets/images/par%20bfs%20pseudo%20%20code.png)

- using bag,putting all  together
	- run time: O(d*V*E)
![par split algo total.png|500](/assets/images/par%20split%20algo%20total.png)

### Data structure: Bag
bag property:
- The data is an unordered collection
- It will allow repetition
- Allow redundant insertions of the same vertex, if necessary

operations:
- traversal
- union, split

##### Pennant
Pennant is: a tree with 2k nodes and a unary root having a child. the child is the root of a complete binary tree. X is the root. Xs is the complete binary tree. So a pennant has **2^n** nodes. Essentially a binary number representation, a binary can be represented as a series of pennants. 

For example, bag of 23 nodes = 2^4 size pennant, 2^2  size pennant, 2^1 size pennant, 2^0 size pennant ($23=10111_2$). Since it is essentially a binary number, 
- it can easily do one element insertion or even two bag union like binary addition. (logn run time)
- logn time for split as well
	- essentially a binary right shift 
	- ![bag split.png|400](/assets/images/bag%20split.png)
