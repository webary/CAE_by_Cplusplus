#pragma once
#ifndef _MATLABFUNCTION_H_
#define _MATLABFUNCTION_H_

#include<cmath>  //exp, floor
#include<cfloat> //FLT_MIN
#include<vector>
#include<iostream>
#include<algorithm>

typedef unsigned uint;
typedef std::vector<float> vectorF;
typedef std::vector<vectorF> vectorF2D;
typedef std::vector<vectorF2D> vectorF3D;
typedef std::vector<vectorF3D> vectorF4D;

//实现matlab中常见操作
namespace mat
{
    //显示一个二维数组
    void disp(vectorF2D &vec);

    //对二维数组vec的按dim维进行原地翻转
    void flip(vectorF2D &vec, unsigned dim);
    //返回四维数组vec按照dim维翻转后的数组
    vectorF4D flip(const vectorF4D  &vec, unsigned dim);

    //返回vec的第dim维的大小(从1开始,最多支持4维)
    template<typename T>
    inline unsigned size(const T &vec, int dim)
    {
        if (dim == 1)
            return vec.size();
        if (dim == 2)
            return vec[0].size();
        if (dim == 3)
            return vec[0][0].size();
        if (dim == 4)
            return vec[0][0][0].size();
        return 0;
    }

    std::vector<unsigned> size(const vectorF3D &vec);

    std::vector<unsigned> size(const vectorF4D &vec);

    void error(const char* str);

    std::vector<int> randperm(unsigned n, unsigned k=0);

    std::vector<int> linspace(int a, int b, unsigned n);

    vectorF zeros(uint a);

    vectorF2D zeros(uint a, uint b);

    vectorF3D zeros(uint a, uint b, uint c);

    vectorF4D zeros(uint a, uint b, uint c, uint d, float first = 0);

    vectorF4D zerosLike(const vectorF4D &vec);

    inline double sigm(double x);
    //将一个Map图每个点加偏置后求sigmoid
    void sigm(vectorF2D &vec, float bias);

    //卷积模式，参考matlab
    enum Shape {
        FULL, SAME, VALID
    };
    vectorF2D conv2(const vectorF2D &A, const vectorF2D &B, Shape shape = FULL);

    //求一个数组的平均值
    float mean(const vectorF &vec, unsigned from, unsigned len);

    //求取一个一维数组中的最大值并返回
    inline float max(const vectorF &vec);

    //求取一个2维数组中的最大值并保存到1维数组中返回
    inline vectorF max(const vectorF2D &vec);

    //求取一个4维数组中的最大值并保存到[1][1][*][*]数组中返回
    vectorF4D max4D(const vectorF4D &vec);

    //将一个4维矩阵重叠复制：[1 1 j k] => [m n j k]
    inline vectorF4D repmat4D(const vectorF4D &vec, int m, int n);

    template<typename T>
    inline bool equal(T a, T b)
    {
        return a - b<1e-5 && b - a<1e-5;
    }

    //把haveMax矩阵中不是maxMax(最大值)的元素置0,即只保留其中的最大值
    vectorF4D reserveMax(vectorF4D &haveMax, const vectorF4D &maxMat);

}

#ifndef _MSC_VER //兼容非项目环境编译（如在CB环境中）
#include"matlabFunc.cpp"
#endif

#endif //_MATLABFUNCTION_H_
