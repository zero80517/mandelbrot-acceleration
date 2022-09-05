#ifndef TYPES_H
#define TYPES_H

typedef unsigned int uint;

struct Params {
    const int halfWidth;
    const int halfHeight;
    const double scaleFactor;
    const double centerX;
    const double centerY;
    const int Limit;
    const int MaxIterations;
    bool *const allBlack;
    bool *const restart;
    bool *const abort;
};

#endif // TYPES_H
