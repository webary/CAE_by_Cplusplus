#include"cae.h"
#include<iostream>
using namespace std;

int main()
{
    CAE cae[3] = {
        { 1, 6, 5, 2, 0 },   //28*28 => 24*24 => 12*12
        { 6, 16, 5, 2, 0 },  //12*12 => 8*8 => 4*4
        { 16, 120, 4, 1, 0 } //4*4 => 1*1
    };

    OPTS opts(2, 1, 0.03, 3);
    InputSet is("trainData1000.txt", 28, 200);

    cae[0].train(is.data, opts); //ph[2 6 12 12]
    cae[0].visualize(is.data);

    auto x1 = cae[0].getCAEOut(is.data);
    cae[1].train(x1, opts); //ph[2 16 4 4]
    cae[1].visualize(x1);
    cout << "查看反演生成的图之后";
    system("pause");

    auto x2 = cae[1].getCAEOut(x1);
    cae[2].train(x2, opts); //ph[2 120 1 1]
    cae[2].visualize(x2);
    return 0;
}
