#include "matlabFunc.h"

//实现matlab中常见操作
namespace mat
{
    //显示一个二维数组
    void disp(vectorF2D &vec)
    {
        unsigned size[2] = { vec.size(), vec[0].size() }, i, j;
        for (i = 0; i < size[0]; ++i) {
            for (j = 0; j < size[1]; ++j)
                std::cout << vec[i][j] << " ";
            std::cout << std::endl;
        }
    }

    //对二维数组vec的按dim维进行原地翻转
    void flip(vectorF2D &vec, unsigned dim)
    {
        unsigned size[2] = { vec.size(), vec[0].size() }, i, j, half;
        if (dim == 1) {
            half = size[0] / 2;
            for (i = 0; i < half; ++i)
                for (j = 0; j < size[1]; ++j)
                    std::swap(vec[i][j], vec[size[0] - i - 1][j]);
        } else if (dim == 2) {
            half = size[1] / 2;
            for (i = 0; i < size[0]; ++i)
                for (j = 0; j < half; ++j)
                    std::swap(vec[i][j], vec[i][size[1] - j - 1]);
        }
    }
    //返回四维数组vec按照dim维翻转后的数组
    vectorF4D flip(const vectorF4D  &vec, unsigned dim)
    {
        if (dim != 1 && dim != 2)
            return vec;
        vectorF4D myVec = vec;
        unsigned size[2] = { myVec.size(), myVec[0].size() }, i, j;
        for (i = 0; i < size[0]; ++i)
            for (j = 0; j < size[1]; ++j)
                flip(myVec[i][j], dim);
        return myVec;
    }

    std::vector<unsigned> size(const vectorF4D &vec)
    {
        std::vector<unsigned> sizeMat(4);
        sizeMat[0] = vec.size();
        sizeMat[1] = vec[0].size();
        sizeMat[2] = vec[0][0].size();
        sizeMat[3] = vec[0][0][0].size();
        return sizeMat;
    }

    void error(const char* str)
    {
        std::cout << "error: " << str << std::endl;
        std::cin.get();
        exit(1);
    }

    std::vector<int> randperm(unsigned n)
    {
        std::vector<int> randp(n);
        for (unsigned i = 0; i < n; ++i)
            randp[i] = i;
        std::random_shuffle(randp.begin(), randp.end()); //algorithm
        return randp;
    }

    vectorF zeros(uint a)
    {
        return vectorF(a, 0);
    }

    vectorF2D zeros(uint a, uint b)
    {
        return vectorF2D(a, vectorF(b, 0));
    }

    vectorF3D zeros(uint a, uint b, uint c)
    {
        return vectorF3D(a, vectorF2D(b, vectorF(c, 0)));
    }

    vectorF4D zeros(uint a, uint b, uint c, uint d, float first/*=0*/)
    {
        return vectorF4D(a, vectorF3D(b, vectorF2D(c, vectorF(d, first))));
    }

    vectorF4D zeros(const vectorF4D vec)
    {
        return vectorF4D(vec.size(), vectorF3D(vec[0].size(), vectorF2D(vec[0][0].size(), vectorF(vec[0][0][0].size(), 0))));
    }

    double sigm(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    }
    //将一个Map图每个点加偏置后求sigmoid
    void sigm(vectorF2D &vec, float bias)
    {
        for (uint i = 0; i<vec.size(); ++i)
            for (uint j = 0; j<vec[i].size(); ++j)
                vec[i][j] = (float)sigm(vec[i][j] + bias);
    }

    vectorF2D conv2(const vectorF2D &A, const vectorF2D &B, Shape shape /*=FULL*/)
    {
        unsigned sizeA[2] = {A.size(), A[0].size()}, sizeB[2] = {B.size() ,B[0].size()};
        vectorF2D result = zeros(sizeA[0] + sizeB[0] - 1, sizeA[1] + sizeB[1] - 1);
        unsigned i, j, m, n;
        for (i = 0; i < sizeA[0]; ++i)
            for (j = 0; j < sizeA[1]; ++j)
                for (m = 0; m < sizeB[0]; ++m)
                    for (n = 0; n < sizeB[1]; ++n)
                        result[i + m][j + n] += A[i][j] * B[m][n];
        if (shape == FULL)
            return result;
        unsigned delta[2] = { 0 }; //x,y方向的偏移量
        unsigned newSize[2] = { 0 }; //新大小
        if (shape == SAME) {
            delta[0] = sizeB[0] / 2;
            delta[1] = sizeB[1] / 2;
            newSize[0] = sizeA[0];
            newSize[1] = sizeA[1];
        } else if (shape == VALID) {
            delta[0] = sizeB[0] - 1;
            delta[1] = sizeB[1] - 1;
            newSize[0] = sizeA[0] - sizeB[0] + 1;
            newSize[1] = sizeA[1] - sizeB[1] + 1;
        }
        vectorF2D res = zeros(newSize[0], newSize[1]);
        for (i = 0; i < newSize[0]; ++i)
            for (j = 0; j < newSize[1]; ++j)
                res[i][j] = result[i + delta[0]][j + delta[1]];
        return res;
    }

    //求一个数组从from开始的len个数的平均值
    float mean(const vectorF &vec, unsigned from, unsigned len)
    {
        if (vec.size() == 0 || len < 0)
            return 0;
        float meanValue = 0;
        for (unsigned i = 0; i < len; ++i)
            meanValue += vec[from + i];
        return meanValue / len;
    }

    //求取一个一维数组中的最大值并返回
    inline float max(const vectorF &vec)
    {
        unsigned v_size = vec.size();
        if (v_size < 1)
            return 0;
        float maxValue = vec[0];
        for (unsigned i = 1; i < v_size; ++i)
            if (maxValue < vec[i])
                maxValue = vec[i];
        return maxValue;
    }

    //求取一个2维数组中的最大值并保存到1维数组中返回
    inline vectorF max(const vectorF2D &vec)
    {
        vectorF maxMat = vec[0];
        unsigned v_size[2] = { vec.size(), vec[0].size() };
        for (unsigned i = 1; i < v_size[0]; ++i)
            for (unsigned j = 0; j < v_size[1]; ++j)
                if (maxMat[j] < vec[i][j])
                    maxMat[j] = vec[i][j];
        return maxMat;

    }

    //求取一个4维数组中的最大值并保存到[*][*][1][1]数组中返回
    vectorF4D max4D(const vectorF4D &vec)
    {
        unsigned i, j, m, n;
        const std::vector<unsigned> v_size = mat::size(vec);
        vectorF4D maxMat = zeros(v_size[0], v_size[1], 1, 1, FLT_MIN);
        for (i = 0; i < v_size[0]; ++i)
            for (j = 0; j < v_size[1]; ++j)
                for (m = 0; m < v_size[2]; ++m)
                    for (n = 0; n < v_size[3]; ++n)
                        if (maxMat[i][j][0][0] < vec[i][j][m][n])
                            maxMat[i][j][0][0] = vec[i][j][m][n];
        return maxMat;

    }

    //将一个4维矩阵重叠复制：[1 1 j k] => [m n j k]
    inline vectorF4D repmat4D(const vectorF4D &vec, int m, int n)
    {
        unsigned i, j;
        const std::vector<unsigned> v_size = mat::size(vec);
        vectorF4D repMat = zeros(m, n, v_size[2], v_size[3]);
        for (i = 0; i < v_size[0]; ++i)
            for (j = 0; j < v_size[1]; ++j)
                repMat[i][j] = vec[0][0];
        return repMat;
    }

    //把haveMax矩阵中不是maxMax(最大值)的元素置0,即只保留其中的最大值
    vectorF4D reserveMax(vectorF4D &haveMax, const vectorF4D &maxMat)
    {
        unsigned i, j, m, n;
        vectorF4D mask(mat::zeros(haveMax)); //最大值的位置用1表示
        const std::vector<unsigned> v_size = mat::size(haveMax);
        for (i = 0; i < v_size[0]; ++i)
            for (j = 0; j < v_size[1]; ++j)
                for (m = 0; m < v_size[2]; ++m)
                    for (n = 0; n < v_size[3]; ++n)
                        if (equal(haveMax[i][j][m][n], maxMat[i][j][0][0]))
                            mask[i][j][m][n] = 1;  //相等则置该位置的mask为1
                        else
                            haveMax[i][j][m][n] = 0; //不相等则把原始数据中该位置置0
        return mask;
    }

}
