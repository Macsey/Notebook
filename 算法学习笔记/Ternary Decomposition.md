# AT_arc164_a [ARC164A] Ternary Decomposition

## 题目描述

给定两个整数 $N,K$，判断 $N$ 能不能用 $K$ 个 $3^{n}(n\in \N)$ 之和表示。

**本题有多组测试数据。**

Translated by [lelml](/user/372622).

## 输入格式

第一行一个整数 $T$，表示数据的组数。

接下来共 $T$ 行，每一行有两个整数 $N,K$。

## 输出格式

共 $T$ 行，每一行表示一组数据判断的结果，如果可以输出 ```Yes```，否则输出 ```No```。

## 输入输出样例 #1

### 输入 #1

```
4
5 3
17 2
163 79
1000000000000000000 1000000000000000000
```

### 输出 #1

```
Yes
No
Yes
Yes
```

## 说明/提示

$1\le T \le 10^5,1 \le K \le N \le 10^{18}$

# AC代码
```cpp
#include<iostream>
#include<algorithm>
#include<string>
#include<map>
#define ll long long 
using namespace std;
int main()
{
	ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
	ll t;
	cin >> t;
	while (t--) {
		ll n, k;
		cin >> n >> k;
		ll temp = n;
		ll count = 0;
		if (k > n) {
			cout << "No" << '\n';
			}
		else if (k == n) {
				cout << "Yes" << '\n';
				continue;
		}
		else {
			while (temp) {
				count += temp % 3;
				temp /= 3;
				
			}
			if (count > k || (k - count) % 2 == 1) {
				cout << "No" << '\n';
			}
			else {
				cout << "Yes" << '\n';
			}
		}
		
		
	}
	return 0;
	}
```
# 核心思路
将数字转化为三进制