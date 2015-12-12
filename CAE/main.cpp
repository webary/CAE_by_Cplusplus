#include"cae.h"
#include<time.h>
using namespace std;

int main()
{
	int t1, t2, i, j;
	vectorF ra;
	t1 = clock();
	for (j = 0; j < 1000; ++j) {
		ra = mat::zeros(10000);
		for (i = 0; i < ra.size(); ++i)
			ra[i] = CAE::randDouble();
	}
	cout << clock() - t1 << endl;


	CAE cae[3];
	cae[0].setup(1, 6, 5, 2, 0);
	cae[1].setup(6, 16, 5, 2, 0);
	cae[2].setup(16, 120, 4, 1, 0);

	OPTS opts(2, 1, 0.03, 10);
	InputSet x(mat::zeros(300, 1, 28, 28), 1);

	cae[0].train(x.data, opts);
	cae[1].train(x.data, opts);//
	cae[2].train(x.data, opts);//

	return 0;
}