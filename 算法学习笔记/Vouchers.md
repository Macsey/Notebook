# AT_abc308_f [ABC308F] Vouchers

## 题目描述

你打算在商店购买 $N$ 件商品。第 $i$ 件商品的定价为 $P_i$ 日元。

你还拥有 $M$ 张优惠券。使用第 $i$ 张优惠券时，可以选择一件定价不低于 $L_i$ 日元的商品，并以比定价低 $D_i$ 日元的价格购买该商品。

每张优惠券只能使用一次，且不能对同一商品叠加使用多张优惠券。

未使用优惠券的商品需按定价购买。请你求出购买全部 $N$ 件商品所需的最小金额。

## 输入格式

输入以如下格式从标准输入读入。

> $N$ $M$ $P_1$ $\ldots$ $P_N$ $L_1$ $\ldots$ $L_M$ $D_1$ $\ldots$ $D_M$

## 输出格式

请输出一个整数，表示答案。

## 输入输出样例 #1

### 输入 #1

```
3 3
4 3 1
4 4 2
2 3 1
```

### 输出 #1

```
4
```

## 输入输出样例 #2

### 输入 #2

```
10 5
9 7 1 5 2 2 5 5 7 6
7 2 7 8 2
3 2 4 1 2
```

### 输出 #2

```
37
```

## 说明/提示

## 限制条件

- $1 \leq N, M \leq 2 \times 10^5$
- $1 \leq P_i \leq 10^9$
- $1 \leq D_i \leq L_i \leq 10^9$
- 所有输入的数值均为整数

## 样例解释 1

考虑将第 $2$ 张优惠券用于第 $1$ 件商品，将第 $3$ 张优惠券用于第 $2$ 件商品。此时，第 $1$ 件商品可用 $4-3=1$ 日元购得，第 $2$ 件商品可用 $3-1=2$ 日元购得，第 $3$ 件商品以 $1$ 日元购得，因此总共需要 $1+2+1=4$ 日元购得全部商品。

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
struct node {
	ll l;
	ll dis;
}dis[200001];
 bool cmp(node a, node b) {
 return a.l < b.l;
 }

int main()
{
	int n, m;
	cin >> n >> m;
	vector<ll>p(n);
	for (int i = 0; i < n; i++) {
		cin >> p[i];
	}
	for (int i = 0; i < m; i++) {
		cin >> dis[i].l;
	}
	for (int i = 0; i < m; i++) {
		cin >> dis[i].dis;
	}
	sort(p.begin(), p.end());
	sort(dis, dis+m,cmp);
	ll sum = 0;
	int t = 0;priority_queue<ll>q;
	for (int i = 0; i < n; i++)
	{
		ll price = p[i];
		while (t < m && dis[t].l <= price) {
			q.push(dis[t].dis);
            t++;
		}
		/*for (int i = 0; i < m; i++) {
			if (dis[i].first <= price) {
				q.push(dis[i].second);
			}
		}*/
		sum += price;
		if (!q.empty()) {
			sum -= q.top();
            q.pop();
		}
	}
	cout << sum << endl;



}
```

## 思考
### 核心思想：贪心算法
对于每一件商品，要找出所有符合条件的优惠券，再找到所有能用的优惠卷中折扣力度最大的，pop ;这时就用到升序，**保证之后的商品都可以用原来没用过的优惠券**，
在分析问题时，注意用到[[优先队列]]，可以快速找到最大值和删去最大值
要注意使用while循环：一定要注意，变量的[[生命周期]]！