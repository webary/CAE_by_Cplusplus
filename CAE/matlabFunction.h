#pragma once
#ifndef _MATLABFUNCTION_H_
#define _MATLABFUNCTION_H_

#include<ctime>
#include<vector>
#include<iostream>
#include<algorithm>
using namespace std;

typedef unsigned uint;
typedef vector<float> vectorF;
typedef vector<vectorF> vectorF2D;
typedef vector<vector<vectorF2D> > vectorF4D;

//实现matlab中常见操作
namespace mat {
	//对二维数组vec的按dim维进行原地翻转
	void flip(vectorF2D &vec, unsigned dim)
	{
		unsigned size[2] = { vec.size(), vec[0].size() }, i, j, half;
		if (dim == 1) {
			half = size[0] / 2;
			for (i = 0; i < half; ++i)
				for (j = 0; j < size[1]; ++j)
					swap(vec[i][j], vec[size[0] - i - 1][j]);
		}
		else if (dim == 2) {
			half = size[1] / 2;
			for (i = 0; i < size[0]; ++i)
				for (j = 0; j < half; ++j)
					swap(vec[i][j], vec[i][size[1] - j - 1]);
		}
	}

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

	//返回vec的第dim维的大小(从1开始,最多支持4维)
	template<typename T>
	inline int size(const T &vec, int dim) {
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

	//当没有传入维度时返回vec的所有维度信息
	template<typename T>
	inline vector<int> size(const T &vec) {
		vector<int> sizeMat(vec.size());
		unsigned i = 0;
		sizeMat[i++] = vec.size();
		if (vec.size() > 1)
			sizeMat[i++] = vec[0].size();
		if (vec.size() > 2)
			sizeMat[i++] = vec[0][0].size();
		if (vec.size() > 3)
			sizeMat[i++] = vec[0][0][0].size();
		return sizeMat;
	}

	inline void error(const char* str) {
		cout << str << endl;
		exit(1);
	}

	inline vector<int> randperm(unsigned n)
	{
		vector<int> randp(n);
		for (unsigned i = 0; i < n; ++i)
			randp[i] = i;
		random_shuffle(randp.begin(), randp.end()); //algorithm
		return randp;
	}

	template<typename T>
	inline vector<T> linspace(const T &a, const T &b, unsigned n)
	{
		vector<T> randp(n);
		T step = (b - a) / (n - 1);
		for (unsigned i = 0; i < n; ++i)
			randp[i] = a + i * step;
		random_shuffle(randp.begin(), randp.end());
		return randp;
	}

	inline vectorF zeros(int a) {
		return vectorF(a, 0);
	}

	inline vectorF2D zeros(int a, int b) {
		return vectorF2D(a, vectorF(b, 0));
	}

	inline vector<vectorF2D> zeros(int a, int b, int c) {
		return vector<vectorF2D>(a, vectorF2D(b, vectorF(c, 0)));
	}

	inline vectorF4D zeros(int a, int b, int c, int d) {
		return vectorF4D(a, vector<vectorF2D>(b, vectorF2D(c, vectorF(d, 0))));
	}

	//按照vec的规格产生一个零矩阵
	template<typename T>
	inline T zeros(const vector<int> &vec) {
		switch (vec.size()) {
		case 4: return zeros(vec[0], vec[1], vec[2], vec[3]); break;
		case 3: return zeros(vec[0], vec[1], vec[2]); break;
		case 2: return zeros(vec[0], vec[1]); break;
		default: return zeros(vec[0]); break;
		}
	}

	inline double sigm(double x) {
		return 1.0 / (1.0 + exp(-x));
	}
	//将一个Map图每个点加偏置后求sigmoid
	inline void sigm(vectorF2D &vec, float bias)
	{
		for (uint i = 0; i<vec.size(); ++i)
			for (uint j = 0; j<vec[i].size(); ++j)
				vec[i][j] = (float)sigm(vec[i][j] + bias);
	}
	//卷积模式，参考matlab
	enum Shape {
		FULL, SAME, VALID
	};
	vectorF2D conv2(const vectorF2D &A, const vectorF2D &B, Shape shape = FULL) {
		unsigned sizeA[2]{ A.size() ,A[0].size() }, sizeB[2] = { B.size() ,B[0].size() };
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
		if (shape == SAME)
		{
			delta[0] = sizeB[0] / 2;
			delta[1] = sizeB[1] / 2;
			newSize[0] = sizeA[0];
			newSize[1] = sizeA[1];
		}
		else if (shape == VALID)
		{
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

	float mean(const vectorF &vec) {
		float meanValue = 0;
		for (unsigned i = 0; i < vec.size(); ++i)
			meanValue += vec[i];
		return meanValue / vec.size();
	}

	//求取一个一维数组中的最大值并返回
	inline float max(const vector<float> &vec) {
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
	inline vectorF max(const vectorF2D &vec) {
		vectorF maxMat = vec[0];
		unsigned v_size[2] = { vec.size(), vec[0].size() };
		for (unsigned i = 1; i < v_size[0]; ++i)
			for (unsigned j = 0; j < v_size[1]; ++j)
				if (maxMat[j] < vec[i][j])
					maxMat[j] = vec[i][j];
		return maxMat;

	}

	//求取一个4维数组中的最大值并保存到2维数组中返回
	inline vectorF2D max(const vectorF4D &vec) {
		vectorF2D maxMat = vec[0][0];
		vector<int> v_size = mat::size(vec);
		for (unsigned i = 1; i < v_size[0]; ++i)
			for (unsigned j = 0; j < v_size[1]; ++j)
				for (unsigned m = 0; m < v_size[2]; ++m)
					for (unsigned n = 0; n < v_size[3]; ++n)
						if (maxMat[m][n] < vec[i][j][m][n])
							maxMat[m][n] = vec[i][j][m][n];
		return maxMat;

	}

	//将一个矩阵重叠复制
	inline vectorF2D repmat(const vectorF2D &vec, int m, int n) {

	}

}
#endif //_MATLABFUNCTION_H_