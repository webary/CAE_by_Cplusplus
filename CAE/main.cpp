#include"cae.h"
#include<time.h>
using namespace std;

int main()
{
	CAE cae[3];
	cae[0].setup(1, 6, 5, 2, 0);
	cae[1].setup(6, 16, 5, 2, 0);
	cae[2].setup(16, 120, 4, 1, 0);

	OPTS opts(2, 1, 0.03, 3);
	InputSet x(mat::zeros(300, 1, 28, 28), 1);

	cae[0].train(x.data, opts);
	cae[1].train(cae[0].getPh(), opts);
	cae[2].train(cae[1].getPh(), opts);

	return 0;
}