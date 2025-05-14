# include <iostream>
# include <string>
# include <cstring>
using namespace std;


//! 模运算满足分配律和结合律，即
//* (a + b) % p = (a % p + b % p) % p
//* (a * b) % p = ((a % p) * (b % p)) % p
//! 因此在矩阵乘法的每一步都取模，最终结果与先计算完整乘积再取模是等价的，但前者避免了中间数过大



//! 在 C/C++ 中，传递二维数组给函数时，第一维（行数）可以省略，
//! 但第二维（列数）必须是编译时常量，以便编译器知道每行元素在内存中的偏移量


const int MAXN = 10; //! 矩阵最大维度
const int MOD = 1e9 + 7; //! 模数


//! 矩阵乘法：C = A * B
//! A尺寸 m x k，B尺寸 k x n，C尺寸 m x n
void matMul (int m, int k, int n, int A[][MAXN], int B[][MAXN], int C[][MAXN]) {
    int temp[MAXN][MAXN]; //! 临时矩阵
    memset(temp, 0, sizeof(temp));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            long long sum = 0;
            for (int p = 0; p < k; p++) {
                sum += (long long)A[i][p] * B[p][j];
            }
            temp[i][j] = sum % MOD;
        }
    }
    //!  复制结果回C
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = temp[i][j];
        }
    }
}

//* 初始化单位矩阵 I，大小 n x n
void setIdentity (int n, int I[][MAXN]) {
    memset(I, 0, sizeof(int) * MAXN * MAXN);
    for (int i = 0; i < n; i++) {
        I[i][i] = 1;
    }
}


//! 矩阵快速幂：计算A^power，A为n x n矩阵，结果存入res
void matrixPower (int n, int A[][MAXN], long long power, int res[][MAXN]) {
    int base[MAXN][MAXN]; //* 临时矩阵
    for (int i = 0; i < n; i++) 
        for (int j = 0; j < n; j++) 
            base[i][j] = A[i][j];

    setIdentity(n, res);  //* res初始化为单位矩阵

    while (power > 0) {
        //! 位运算，判断power是否为奇数
        if (power & 1) {
            matMul(n, n, n, res, base, res); //! A^4 * A = A^5
        } else {
            //! base 的作用是记录当前迭代的矩阵幂
            matMul(n, n, n, base, base, base); //! A^4 * A^4 = A^8
        }
        power >>= 1;
    }
}

int main() {
    int n;
    long long k;
    cin >> n >> k;

    int A[MAXN][MAXN];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[i][j];
        }
    }

    int res[MAXN][MAXN];
    matrixPower(n, A, k, res);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << res[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}