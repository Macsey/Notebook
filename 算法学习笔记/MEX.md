# AT_abc308_e [ABC308E] MEX

## 题目描述

给定一个由 $0,1,2$ 组成的长度为 $N$ 的数列 $A=(A_1,A_2,\dots,A_N)$，以及一个由 `M`、`E`、`X` 组成的长度为 $N$ 的字符串 $S=S_1S_2\dots S_N$。

请计算所有满足 $1 \leq i < j < k \leq N$ 且 $S_iS_jS_k=$ `MEX` 的整数三元组 $(i,j,k)$，对于每个三元组，求 $\text{mex}(A_i,A_j,A_k)$ 的总和。这里，$\text{mex}(A_i,A_j,A_k)$ 表示不等于 $A_i,A_j,A_k$ 的最小非负整数。

## 输入格式

输入通过标准输入给出，格式如下：

> $N$ $A_1$ $A_2$ $\dots$ $A_N$ $S$

## 输出格式

请输出答案，结果为一个整数。

## 输入输出样例 #1

### 输入 #1

```
4
1 1 0 2
MEEX
```

### 输出 #1

```
3
```

## 输入输出样例 #2

### 输入 #2

```
3
0 0 0
XXX
```

### 输出 #2

```
0
```

## 输入输出样例 #3

### 输入 #3

```
15
1 1 2 0 0 2 0 2 0 0 0 0 0 2 2
EXMMXXXEMEXEXMM
```

### 输出 #3

```
13
```

## 说明/提示

## 限制条件

- $3 \leq N \leq 2 \times 10^5$
- $N$ 为整数
- $A_i \in \{0,1,2\}$
- $S$ 是由 `M`、`E`、`X` 组成的长度为 $N$ 的字符串

## 样例解释 1

满足 $S_iS_jS_k = $ `MEX` 的 $(i,j,k)$ 共有 $(1,2,4),(1,3,4)$ 两组。$\text{mex}(A_1,A_2,A_4)=\text{mex}(1,1,2)=0$，$\text{mex}(A_1,A_3,A_4)=\text{mex}(1,0,2)=3$，因此答案为 $0+3=3$。

由 ChatGPT 4.1 翻译
# AC代码
```cpp
#include<iostream>
#include<algorithm>
#include<vector>
#include<string>
#include<queue>
#define ll long long
using namespace std;
ll m[200001][3];
ll x[200001][3];
int mex(int x,int y,int z) {
	int a[3] = { x,y,z };
	sort(a, a + 3);
	for (int i = 0; i < 3; i++) {
		int count = 0;
		for (int j = 0; j < 3; j++) {
			if (a[j] != i) {
				count++;
			}
		}
		if (count == 3) {
			return i;
		}
	}
	return 3;
}
int main()
{
	ll n;
	cin >> n;
	vector<int>a(n + 1);
	for (int i = 1; i <= n; i++) {
		cin >> a[i];
	}
	string s;
	cin >> s;
	for (int i = 1; i <= n; i++) {
		for (int j = 0; j < 3; j++) {
				m[i][j] = m[i - 1][j];
				x[i][j] = x[i - 1][j];
		}
		if (s[i-1] == 'M') {
			m[i][a[i]]++;
		}
		else if (s[i - 1] == 'X') {
			x[i][a[i]]++;
		}
	}
	unsigned ll sum = 0;
    for (int i = 1; i <= n; i++) { 
		if (s[i-1]=='E') {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					sum += m[i][j] * (x[n][k] - x[i][k])*mex(j, a[i], k);
				}
			}
		}
	}
	cout << sum << endl;
	return 0;
}
```

# 核心思想
考虑复杂度
三层循环遍历15次方超时，考虑前缀和，遍历E，统计之前出现的M中0,1,2的次数，之后出现的X中0,1,2的次数 
运用组合数学，
```cpp
sum += m[i][j] * (x[n][k] - x[i][k])*mex(j, a[i], k);
```