#include <cmath>

double arithmetic_mean(double *v, int len) {

    double mu = 0.0;
    for (int i = 0; i < len; i++)
        mu += (double)v[i];
    mu /= (double)len;

    return (mu);
}

float geometric_mean(float *v, int len) {

    double mu = 1.0;
    for (int i = 0; i < len; i++) {
        mu *= (v[i] > 0) ? ((double)v[i]) : 1;
    }
    mu = pow(mu, 1.0 / len);

    return (mu);
}
