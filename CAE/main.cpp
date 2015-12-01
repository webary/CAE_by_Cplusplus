#include"cae.h"
#include<iostream>

void disp(vector<vector<double> > &vec)
{
	unsigned size[2] = { vec.size(), vec[0].size() }, i, j;
	for (i = 0; i < size[0]; ++i) {
		for (j = 0; j < size[1]; ++j)
			cout << vec[i][j] << " ";
		cout << endl;
	}
	cout << endl;
}

int main()
{
	CAE cae[3];
	cae[0].setup(1, 6, 5, 2, 0);
	cae[1].setup(6, 16, 5, 2, 0);
	cae[2].setup(16, 120, 4, 1, 0);

	OPTS opts(2, 1, 0.03, 10);
	InputSet x(zeros(300, 1, 28, 28), 1);

	cae[0].train(x.data, opts);
	cae[1].train(x.data, opts);//
	cae[2].train(x.data, opts);//

	return 0;
}