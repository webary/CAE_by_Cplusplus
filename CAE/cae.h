#pragma once
#ifndef _CAE_H_
#define _CAE_H_

#include "matlabFunction.h"

//int batchsize;	bool shuffle;	double alpha;	int numepochs;
struct OPTS {
	int batchsize;
	bool shuffle;
	double alpha;
	int numepochs;

	OPTS() = default;

	OPTS(int bs, bool sh, double al, int ne) {
		batchsize = bs;
		shuffle = sh;
		alpha = al;
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

struct CAE {
	int ic; //input channels
	int oc; //ouput channels
	int ks; //kernel size
	int ps; //pool size
	double noise;
	vectorF b, c, L, db, dc;
	vectorF4D w, w_tilde, h, dh, h_pool, h_mask, ph, out;
	vector<vectorF4D> dw;

	CAE() = default;
	CAE(int _ic, int _oc, int _ks, int _ps, double _noise)
		: b(mat::zeros(_oc)), c(mat::zeros(_ic)),
		  w(_oc, vector<vectorF2D>(_ic, vectorF2D(_ks, vectorF(_ks)))) {
		ic = _ic;
		oc = _oc;
		ks = _ks;
		ps = _ps;
		noise = _noise;

		int i, j, m, n;
		for (i = 0; i < oc; ++i)
			for (j = 0; j < ic; ++j)
				for (m = 0; m < ks; ++m)
					for (n = 0; n < ks; ++n)
						w[i][j][m][n] = (float)(randDouble() - .5) * 60 / (oc*ks*ks);
		w_tilde = mat::flip(mat::flip(w, 1), 2);
	}

	void setup(int _ic, int _oc, int _ks, int _ps, double _noise) {
		*this = CAE(_ic, _oc, _ks, _ps, _noise);
	}

	void train(const vectorF4D &x, const OPTS &opts) {
		PARA para = check(x, opts);
		L = mat::zeros(opts.numepochs * (int)para.bnum);
		int t_start;
		vector<int> idx;
		vectorF4D batch_x = mat::zeros(para.bsze, mat::size(x, 2)
		                               , mat::size(x, 3), mat::size(x, 4));
		for (int i = 0; i < opts.numepochs; ++i) {
			cout << "epoch " << i << "/" << opts.numepochs << endl;
			t_start = clock(); //开始计时
			if (opts.shuffle)
				idx = mat::randperm(para.pnum);
			else
				idx = mat::linspace(1, para.pnum, para.pnum);
			for (int j = 0; j < para.bnum; ++j) {
				for (int k = 0; k < para.bsze; ++k)
					batch_x[k] = x[j * para.bsze + k];
				ffbp(batch_x, para);
				update(opts);
				L[i*(int)para.bnum + j] = 0;// loss;
			}
			//显示平均值
			cout << mat::mean(L) << endl << clock() - t_start << endl;
			t_start = clock();
		}
	}

	void ffbp(const vectorF4D &x, PARA &para) {
		vectorF4D x_noise = x;
		int i, j, m, n;
		int x_size[4];
		for (i = 0; i < 4; ++i)
			x_size[i] = mat::size(x_noise, i + 1);
		for (i = 0; i < x_size[0]; ++i)
			for (j = 0; j < x_size[1]; ++j)
				for (m = 0; m < x_size[2]; ++m)
					for (n = 0; n < x_size[3]; ++n)
						if (randDouble() < noise)
							x_noise[i][j][m][n] = 0;
		up(x_noise, para);
		pool(para);
		down(para);
		grad(x, para);
	}

	void up(const vectorF4D &x, PARA &para) {
		h = mat::zeros(para.bsze, oc, para.m - ks + 1, para.m - ks + 1);
		for (int pt = 0; pt < para.bsze; ++pt)
			for (int _oc = 0; _oc < oc; ++_oc) {
				for (int _ic = 0; _ic < ic; ++_ic)
					addVector(h[pt][_oc], h[pt][_oc], mat::conv2(x[pt][_ic], w[_oc][_ic], mat::VALID));
				mat::sigm(h[pt][_oc], b[_oc]);
			}
	}

	void pool(const PARA &para) {
		if (ps >= 2) {
			h_pool = mat::zeros<vectorF4D>(mat::size(h));
			h_mask = mat::zeros<vectorF4D>(mat::size(h));
			ph = mat::zeros(para.pgrds, para.pgrds, oc, para.bsze);
			vectorF4D grid = mat::zeros(para.bsze, oc, ps, ps);
			vectorF4D sparse_grid, mx, mask, tmpMax;
			for (int i = 0; i < para.pgrds; ++i) {
				for (int j = 0; j < para.pgrds; ++j) {
					for (int pt = 0; pt < para.bsze; ++pt)
						for (int _oc = 0; _oc < oc; ++_oc)
							for (int jj = 0; jj < ps; ++jj)
								for (int ii = 0; ii < ps; ++ii)
									grid[pt][_oc][jj][ii] = h[pt][_oc][j*ps + jj][i*ps + ii];
					tmpMax = mat::max4D(grid);
					ph[i][j] = tmpMax[0][0];
					mx = mat::repmat4D(tmpMax, ps, ps); //用最大值填充
					sparse_grid = grid;
					mask = mat::reserveMax(sparse_grid, tmpMax);
					for (int pt = 0; pt < para.bsze; ++pt)
						for (int _oc = 0; _oc < oc; ++_oc)
							for (int jj = 0; jj < ps; ++jj)
								for (int ii = 0; ii < ps; ++ii) {
									h_pool[pt][_oc][j*ps + jj][i*ps + ii] = sparse_grid[pt][_oc][jj][ii];
									h_mask[pt][_oc][j*ps + jj][i*ps + ii] = mask[pt][_oc][jj][ii];
								}
				}
			}
		} else {
			ph = h;
			h_pool = h;
			h_mask = h;
		}

	}

	void down(PARA &para) {
		out = mat::zeros(para.bsze, ic, para.m, para.m);
		for (int pt = 0; pt < para.bsze; ++pt)
			for (int _ic = 0; _ic < ic; ++_ic) {
				for (int _oc = 0; _oc < oc; ++_oc)
					addVector(h[pt][_ic], h[pt][_ic], mat::conv2(h_pool[pt][_oc], w_tilde[_oc][_ic], mat::FULL));
				mat::sigm(out[pt][_ic], b[_ic]);
			}
	}

	void grad(const vectorF4D &x, PARA &para) {
		//% o = sigmoid(y'), y' = sigma(maxpool(sigmoid(h'))*W~)+c, h' = W*x+b
		//% y', h' are pre-activation terms
		cae.err = (cae.o-x);
cae.loss = 1/2 * sum(cae.err(:) .^2 )/para.bsze;
		//% size(cae.loss)

		//% dloss/dy' = (y-x)(y(1-y))
		cae.dy = cae.err.*(cae.o.*(1-cae.o))/para.bsze;
		//% dloss/dc = sigma(dy')

		cae.dc = reshape(sum(sum(cae.dy)),[size(cae.c) para.bsze]);
		//% dloss/dmaxpool(sigmoid(h')) = sigma(dy'*W)
		cae.dh = zeros(size(cae.h));
for pt = 1:
		         para.bsze
         for oc = 1:
			                  cae.oc
                  for ic = 1:
				                           cae.ic
                           cae.dh(:,:,oc,pt) = cae.dh(:,:,oc,pt)+conv2(cae.dy(:,:,ic,pt),cae.w(:,:,ic,oc),'valid');
		end
		end
		end
		if cae.ps>=2
		cae.dh = cae.dh.*cae.h_mask;
	end
	//% dsigmoid(h')/dh'
	cae.dh = cae.dh.*(cae.h.*(1-cae.h));
		//% dloss/db = sigma(dh')
		if para.pgrds>=2
		cae.db = reshape(sum(sum(cae.dh)),[size(cae.b) para.bsze]);
		else
			cae.db = reshape(cae.dh,[size(cae.b) para.bsze]);
		end
		//% dloss/dw = x~*dh'+dy'~*h
		cae.dw = zeros([size(cae.w) para.bsze]);
		cae.dy_tilde = flip(flip(cae.dy,1),2);
		x_tilde = flip(flip(x,1),2);
for pt = 1:
		         para.bsze
         for oc = 1:
			                  cae.oc
                  for ic = 1:
				                           cae.ic
				                           //% x~*dh+dy~*h, perfect
                           cae.dw(:,:,ic,oc,pt) = conv2(x_tilde(:,:,ic,pt),cae.dh(:,:,oc,pt),'valid')+conv2(cae.dy_tilde(:,:,ic,pt),cae.h_pool(:,:,oc,pt),'valid');
		end
		end
		end

		cae.dc = sum(cae.dc,3);
		cae.db = sum(cae.db,3);
		cae.dw = sum(cae.dw,5);
	}

	void update(const OPTS &opts) {
		b -= opts.alpha * db;  //请先确保db已初始化
		c -= opts.alpha * dc;
		w -= opts.alpha * dw;
		w_tilde = flip(flip(w,1),2);
	}

	PARA check(const vectorF4D &x, const OPTS &opts) {
		PARA para;
		para.m = mat::size(x, 4);
		para.pnum = mat::size(x, 1);
		para.pgrds = (float)(para.m - ks + 1) / ps;
		para.bsze = opts.batchsize;
		para.bnum = (float)para.pnum / para.bsze;
		if (mat::size(x, 2) != ic)
			mat::error("number of input chanels doesn't match.");
		if (ks > para.m)
			mat::error("too large kernel.");
		if (floor(para.pgrds) < para.pgrds)
			mat::error("sides of hidden representations should be divisible by pool size.");
		if (floor(para.bnum) < para.bnum)
			mat::error("number of data points should be divisible by batch size.");
		return para;
	}


	static double randDouble() {
		return rand() / (double)RAND_MAX;
	}
	static double randDouble(double a, double b) {
		return (b - a)*randDouble() + a;
	}
	//把out与vec1相加保存到out
	static void addVector(vectorF2D& output, const vectorF2D& vec1) {
		uint i, j, sizeOut[2] = { output.size() ,output[0].size() };
		uint sizeVec[2] = { vec1.size() ,vec1[0].size() };
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