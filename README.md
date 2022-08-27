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