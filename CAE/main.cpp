#include"cae.h"
using namespace std;

int main()
{
    CAE cae[3];
    cae[0].setup(1, 6, 5, 2, 0);
    cae[1].setup(6, 16, 5, 2, 0);
    cae[2].setup(16, 120, 4, 1, 0);

    OPTS opts(2, 1, 0.03, 3);
    InputSet x("trainData1000.txt", 28, 200);

    cae[0].train(x.data, opts); //ph[2 6 12 12]
    cae[0].visualize(x.data);

    cae[1].train(cae[0].ph, opts); //ph[2 16 4 4]
    cae[2].train(cae[1].ph, opts); //ph[2 120 1 1]

    return 0;
}
