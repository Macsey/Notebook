1. const修饰指针——常量指针
const int * a=&b;
**指针指向可以改**
**但是指针指向的值不可以改**

2. const修饰常量——指针常量
int * const a=&b;
指向的值可以更改
但是指向不可以更改

3. const修饰指针，也修饰常量
const int * const a=&b;
指向和指向的值都不可以改

[[const与结构体]]
