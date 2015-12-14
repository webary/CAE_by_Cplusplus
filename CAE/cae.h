#pragma once
#ifndef _CAE_H_
#define _CAE_H_

#include "matlabFunc.h"

//int batchsize;	bool shuffle;	double alpha;	int numepochs;
struct OPTS {
	int batchsize;
	bool shuffle;
	float alpha;
	int numepochs;

	OPTS() = default;

	OPTS(int bs, bool sh, double al, int ne) {
		batchsize = bs;
		shuffle = sh;
		alpha = (float)al;
		numepochs = ne;
	}
};

struct PARA {
	int m;      //size(x,4)
	int pnum;   //number of data points
	float pgrds;//pool grids
	int bsze;   //batch size
	float bnum; //number of batches
};

struct InputSet {
	//[1000  1  28  28]  注意matlab里是[28 28 1 1000]
	vectorF4D data; //输入图数据
	int tag;	//输入图标签

	InputSet(vectorF4D &_data, int _tag) : data(_data), tag(_tag) {}
};

class CAE {
	int ic; //input channels
	int oc; //ouput channels
	int ks; //kernel size
	int ps; //pool size
	double noise, loss;
	vectorF b, c, L, db, dc;
	vectorF4D w, h, h_pool, h_mask, ph, out, err;
	vectorF4D dw, w_tilde, dh, dy, dy_tilde;
public:
	CAE() = default;
	CAE(int _ic, int _oc, int _ks, int _ps, double _noise);

	void setup(int _ic, int _oc, int _ks, int _ps, double _noise);

	void train(const vectorF4D &x, const OPTS &opts);

	vectorF4D& getPh() {
		return ph;
	}
protected:
	void ffbp(const vectorF4D &x, PARA &para);

	void up(const vectorF4D &x, PARA &para);

	void pool(const PARA &para);

	void down(PARA &para);

	void grad(const vectorF4D &x, PARA &para);

	void update(const OPTS &opts);

	PARA check(const vectorF4D &x, const OPTS &opts);

	//产生一个随机浮点数返回
	static double randFloat() {
		return rand() / (float)RAND_MAX;
	}
	static double randFloat(double a, double b) {
		return (b - a)*randFloat() + a;
	}
	//把out与vec1相加保存到out
	static void addVector(vectorF2D& output, const vectorF2D& vec1) {
		unsigned i, j, sizeOut[2] = { output.size() ,output[0].size() };
		unsigned sizeVec[2] = { vec1.size() ,vec1[0].size() };
		for (i = 0; i < sizeOut[0] && i < sizeVec[0]; ++i)
			for (j = 0; j < sizeOut[1] && j < sizeVec[1]; ++j)
				output[i][j] += vec1[i][j];
	}
	template<typename...Args>
	static void addVector(vectorF2D &out, const vectorF2D &vec1, const Args...args) {
		addVector(out, vec1);
		addVector(out, args...);
	}
};

#endif //_CAE_H_