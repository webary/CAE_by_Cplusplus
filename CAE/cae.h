#pragma once
#ifndef _CAE_H_
#define _CAE_H_

#include<ctime>
#include<string>
#include<fstream>
#include"matlabFunc.h"

//int batchsize;	bool shuffle;	double alpha;	int numepochs;
struct OPTS {
    int batchsize;
    bool shuffle;
    float alpha;
    int numepochs;

    OPTS() = default;

    OPTS(int bs, bool sh, double al, int np)
    {
        batchsize = bs;
        shuffle = sh;
        alpha = (float)al;
        numepochs = np;
    }
};

//size(x,4) | number of data points | pool grids | batch size | number of batches
struct PARA {
    int m;     //size(x,4)
    int pnum;  //number of data points
    int pgrds; //pool grids
    int bsze;  //batch size
    int bnum;  //number of batches
};
/*
vectorF4D data; //输入图数据.[n  1  28  28]
std::vector<int> dataTag;//输入图标签
*/
struct InputSet {
    vectorF4D data; //输入图数据.[n  1  28  28]  注意matlab里是[28 28 1 n]
    std::vector<int> dataTag;//输入图标签

    InputSet() = default;
    InputSet(vectorF4D &_data) : data(_data) {}
    InputSet(const char* file, int size, int len = 0, bool haveTag = 1)
    {
        loadInput(file, size, len, haveTag);
    }

    //数据文件，每组数据的规格，数据组数，是否包含标签
    int loadInput(const char* file, unsigned size, unsigned len = 0, bool haveTag = 1)
    {
        std::ifstream loadFile(file);
        if (loadFile.is_open()) {
            if (len > 0) {
                data.reserve(len);
                if (haveTag)
                    dataTag.reserve(len);
            }
            uint i, j, k, tag;
            bool success = true;  //如果某个数据读取错误，则失败
            vectorF3D input = mat::zeros(1, size, size);
            for (i = 0; len < 1 || i < len; ++i) {
                for (j = 0; j < size && success; ++j)
                    for (k = 0; k < size; ++k)
                        if (!(loadFile >> input[0][j][k])) {
                            success = false;
                            break;
                        }
                //该组数据没有读取完整，则不加入训练集
                if (!success || haveTag && !(loadFile >> tag))
                    break;
                data.push_back(input);
                if (haveTag)
                    dataTag.push_back(tag);
            }
            loadFile.close();
            return i;
        } else
            return 0;
    }
};

class CAE {
    int ic; //input channels
    int oc; //ouput channels
    int ks; //kernel size
    int ps; //pool size
    float noise, loss;
    vectorF b, c, L, db, dc;
    vectorF4D w, h, h_pool, h_mask, out, err;//out是反演之后的结果
    vectorF4D dw, w_tilde, dh, dy, dy_tilde;
public:
    vectorF4D ph; //卷积操作加下采样之后的结果.一维大小为opts.batchsize

    CAE() = default;
    CAE(int _ic, int _oc, int _ks, int _ps, double _noise);

    void setup(int _ic, int _oc, int _ks, int _ps, double _noise);

    enum TarinTestType {
        TT_Train,
        TT_Test,
        TT_None
    };
    void train(const vectorF4D &x, const OPTS &opts, TarinTestType tt_type = TT_Train);

    const vectorF4D& getPh() const
    {
        return ph;
    }

    //randomly selected reconstruction results aside the original input.
    void visualize(const vectorF4D &x, std::string file = "cae_vis.txt");

    //取得数据集x通过CAE后的输出
    vectorF4D getCAEOut(const vectorF4D &x);
protected:
    void ffbp(const vectorF4D &x, PARA &para);
    //重新计算h
    void up(const vectorF4D &x, const PARA &para);

    void pool(const PARA &para);

    void down(PARA &para);

    void grad(const vectorF4D &x, PARA &para);

    void update(const OPTS &opts);

    PARA check(const vectorF4D &x, const OPTS &opts);

    //产生一个随机浮点数返回
    static float randFloat()
    {
        return rand() / (float)RAND_MAX;
    }
    static float randFloat(double a, double b)
    {
        return (float)((b - a)*randFloat() + a);
    }
    //把out与vec1相加保存到out(使用了可变模板函数参数,需c++11支持)
    static void addVector(vectorF2D& output, const vectorF2D& vec1)
    {
        unsigned i, j, sizeOut[2] = { output.size() ,output[0].size() };
        unsigned sizeVec[2] = { vec1.size() ,vec1[0].size() };
        for (i = 0; i < sizeOut[0] && i < sizeVec[0]; ++i)
            for (j = 0; j < sizeOut[1] && j < sizeVec[1]; ++j)
                output[i][j] += vec1[i][j];
    }
    template<typename...Args>
    static void addVector(vectorF2D &out, const vectorF2D &vec1, const Args...args)
    {
        addVector(out, vec1);
        addVector(out, args...);
    }
    //仅设置一次随机数种子
    static void setSrand()
    {
        static bool first = 1;
        if (first) {
            srand(unsigned(time(NULL)));
            first = 0;
        }
    }
};

#ifndef _MSC_VER //兼容非项目环境编译（如在CB环境中）
#include"cae.cpp"
#endif

#endif //_CAE_H_
