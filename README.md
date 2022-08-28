**Content**

- [mandelbrot-acceleration](#mandelbrot-acceleration)
- [Questions](#questions)

# mandelbrot-acceleration

Acceleration of evaluation colors in [Qt mandelbrot example](https://doc.qt.io/qt-6/qtcore-threads-mandelbrot-example.html#:~:text=The%20Mandelbrot%20example%20demonstrates%20multi,the%20world's%20most%20famous%20fractal.).

# Questions

1. Why ```reinterpret_cast``` break runtime?

    ```cpp
    scanLine = reinterpret_cast<uint *>(image.scanLine(y + halfHeight));
    ```

    So I used ```pragma omp critical```, and problem was solved

    ```cpp
    #pragma omp critical
    {
        scanLine = reinterpret_cast<uint *>(image.scanLine(y + halfHeight));
    }
    ```

    **UPD:** now openmp example is used [bits()](https://doc.qt.io/archives/qt-4.8/qimage.html#bits)  to get pixel data instead of [scanLine()](https://doc.qt.io/archives/qt-4.8/qimage.html#scanLine). But the question is still actual...