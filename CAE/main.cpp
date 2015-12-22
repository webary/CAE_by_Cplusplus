#include"cae.h"
using namespace std;

int main()
{
    CAE cae[3];
    cae[0].setup(1, 6, 5, 2, 0);
    cae[1].setup(6, 16, 5, 2, 0);
    cae[2].setup(16, 120, 4, 1, 0);

    OPTS opts(2, 1, 0.03, 2);
    InputSet x("trainData.txt", 28, 300);

    cae[0].train(x.data, opts);
    cae[1].train(cae[0].getPh(), opts);
    cae[2].train(cae[1].getPh(), opts);

    return 0;
}
