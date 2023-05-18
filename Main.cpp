#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <random>
class ComplexNumber {
public:
    ComplexNumber(double real = 0.0, double imaginary = 0.0)
        : real_(real), imaginary_(imaginary) {}

    double getReal() const { return real_; }
    double getImaginary() const { return imaginary_; }
    void setReal(double real) { real_ = real; }
    void setImaginary(double imaginary) { imaginary_ = imaginary; }

    ComplexNumber operator+(const ComplexNumber& other) const {
        return ComplexNumber(real_ + other.real_, imaginary_ + other.imaginary_);
    }

    ComplexNumber operator-(const ComplexNumber& other) const {
        return ComplexNumber(real_ - other.real_, imaginary_ - other.imaginary_);
    }

    ComplexNumber operator*(const ComplexNumber& other) const {
        double real = real_ * other.real_ - imaginary_ * other.imaginary_;
        double imaginary = real_ * other.imaginary_ + imaginary_ * other.real_;
        return ComplexNumber(real, imaginary);
    }

    ComplexNumber square() const {
        return ComplexNumber(real_ * real_ - imaginary_ * imaginary_, 2.0 * real_ * imaginary_);
    }

private:
    double real_;
    double imaginary_;
};

class MandelbrotSet {
public:
    MandelbrotSet() = default;
    MandelbrotSet(int width, int height, int maxIterations, double xMin, double xMax, double yMin, double yMax)
        : width_(width), height_(height), maxIterations_(maxIterations), xMin_(xMin), xMax_(xMax), yMin_(yMin), yMax_(yMax) {
        mandelbrotData_.resize(height_, std::vector<cv::Vec3b>(width_));
    }

    void setData(int width, int height, int maxIterations, double xMin, double xMax, double yMin, double yMax)
    {
        width_ = width;
        height_ = height;
        maxIterations_ = maxIterations;
        xMin_ = xMin;
        xMax_ = xMax;
        yMin_ = yMin;
        yMax_ = yMax;
        mandelbrotData_.resize(height_, std::vector<cv::Vec3b>(width_));
    }

    int getIterationsSimple(long double a, long double b) {

        // Apply the Mandelbrot algorithm to the point
        long double ca = a;
        long double cb = b;
        int iterations = 0;
        while (iterations < 1000) {
            long double aa = a * a - b * b;
            long double bb = 2 * a * b;
            a = aa + ca;
            b = bb + cb;
            if (a * a + b * b > 2.0) {
                break;
            }
            ++iterations;
        }
        return iterations;
    }

    int getIterations(long double real, long double imaginary) {

        ComplexNumber c(real, imaginary);
        ComplexNumber z;

        int iterations = 0;
        while (iterations < maxIterations_) {
            z = z.square() + c;
            if (z.getReal() * z.getReal() + z.getImaginary() * z.getImaginary() > 4.0) {
                break;
            }
            iterations++;
        }
        return iterations;
    }

    void calculateMandelbrotSet1() {
        std::vector<std::thread> threads;
        std::mutex mutex;

        int numThreads = std::thread::hardware_concurrency();
        int rowsPerThread = height_ / numThreads;

        for (int i = 0; i < numThreads; ++i) {
            int startRow = i * rowsPerThread;
            int endRow = (i == numThreads - 1) ? height_ : startRow + rowsPerThread;
            threads.emplace_back([this, startRow, endRow, &mutex]() {
                for (int y = startRow; y < endRow; ++y) {
                    for (int x = 0; x < width_; ++x) {
                        double real = mapToRange(x, 0, width_ - 1, xMin_, xMax_);
                        double imaginary = mapToRange(y, 0, height_ - 1, yMin_, yMax_);

                        int iterations = getIterationsSimple(real, imaginary);
                        mandelbrotData_[y][x] = getColor(iterations);
                    }
                }
                std::lock_guard<std::mutex> lock(mutex);
                completedRows_ += endRow - startRow;
                if (completedRows_ % (height_ / 10) == 0) {
                    std::cout << "Progress: " << (completedRows_ * 100) / height_ << "%" << std::endl;
                }
                });
        }

        // Wait for all threads to finish
        for (auto& thread : threads) {
            thread.join();
        }
    }

    void calculateMandelbrotSet() {
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                long double real = mapToRange(x, 0, width_ - 1, xMin_, xMax_);
                long double imaginary = mapToRange(y, 0, height_ - 1, yMin_, yMax_);

                ComplexNumber c(real, imaginary);
                ComplexNumber z;

                int iterations = 0;
                while (iterations < maxIterations_) {
                    z = z.square() + c;
                    if (z.getReal() * z.getReal() + z.getImaginary() * z.getImaginary() > 4.0) {
                        break;
                    }
                    iterations++;
                }

                mandelbrotData_[y][x] = getColor(iterations);
            }
        }
    }

    void calculateMandelbrotSetThread() {
        std::vector<std::thread> threads;
        std::mutex mutex;

        int numThreads = std::thread::hardware_concurrency();
        numThreads = 24;
        int rowsPerThread = height_ / numThreads;

        for (int i = 0; i < numThreads; ++i) {
            int startRow = i * rowsPerThread;
            int endRow = (i == numThreads - 1) ? height_ : startRow + rowsPerThread;
            threads.emplace_back([this, startRow, endRow, &mutex]() {
                for (int y = startRow; y < endRow; ++y) {
                    for (int x = 0; x < width_; ++x) {
                        double real = mapToRange(x, 0, width_ - 1, xMin_, xMax_);
                        double imaginary = mapToRange(y, 0, height_ - 1, yMin_, yMax_);

                        ComplexNumber c(real, imaginary);
                        ComplexNumber z;

                        int iterations = 0;
                        while (iterations < maxIterations_) {
                            z = z.square() + c;
                            if (z.getReal() * z.getReal() + z.getImaginary() * z.getImaginary() > 4.0) {
                                break;
                            }
                            iterations++;
                        }
                        mandelbrotData_[y][x] = getColor(iterations);
                    }
                }
                std::lock_guard<std::mutex> lock(mutex);
                completedRows_ += endRow - startRow;
                if (completedRows_ % (height_ / 10) == 0) {
                    std::cout << "Progress: " << (completedRows_ * 100) / height_ << "%" << std::endl;
                }
            });
        }

        // Wait for all threads to finish
        for (auto& thread : threads) {
            thread.join();
        }
    }


    const std::vector<std::vector<cv::Vec3b>>& getMandelbrotData() const {
        return mandelbrotData_;
    }

private:
    int width_;
    int height_;
    int maxIterations_;
    double xMin_;
    double xMax_;
    double yMin_;
    double yMax_;
    std::vector<std::vector<cv::Vec3b>> mandelbrotData_;
    int completedRows_ = 0;


    double mapToRange(double value, double inputMin, double inputMax, double outputMin, double outputMax) {
        return outputMin + ((value - inputMin) / (inputMax - inputMin)) * (outputMax - outputMin);
    }

    cv::Vec3b getColor(int iterations) {
        double t = static_cast<double>(iterations) / static_cast<double>(1000);
        int r = static_cast<int>(9 * (1 - t) * t * t * t * 255);
        int g = static_cast<int>(15 * (1 - t) * (1 - t) * t * t * 255);
        int b = static_cast<int>(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
        return cv::Vec3b(r, g, b);
    }

};

class ImageGenerator {
public:
    ImageGenerator()= default;
    ImageGenerator(int width, int height) : width_(width), height_(height) {}
    void generateImage(const std::vector<std::vector<cv::Vec3b>>& mandelbrotData) {
        image_ = cv::Mat(height_, width_, CV_8UC3);
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                image_.at<cv::Vec3b>(y, x) = mandelbrotData[y][x];
            }
        }
    }

    void displayImage() {
        cv::imshow("Mandelbrot Set", image_);
        //cv::waitKey(0);

    }
private:
    int width_;
    int height_;
    cv::Mat image_;
};

class UserInterface {
public:
    UserInterface(int width, int height, double xMin, double xMax, double yMin, double yMax)
        : width_(width), height_(height), xMin_(xMin), xMax_(xMax), yMin_(yMin), yMax_(yMax) {
        
        cv::namedWindow("Mandelbrot Set");
        cv::setMouseCallback("Mandelbrot Set", &UserInterface::onMouse,this);
    }

    UserInterface(MandelbrotSet& mandelbrotSet, ImageGenerator& imageGenerator)
    {
        cv::namedWindow("Mandelbrot Set");
        cv::setMouseCallback("Mandelbrot Set", &UserInterface::onMouse, this);

        mandelbrotSet_ = mandelbrotSet;
        imageGenerator_ = imageGenerator;
    }
        
        static void onMouse(int event, int x, int y, int flags, void* userdata)
        {
            UserInterface* ui = static_cast<UserInterface*>(userdata);
            if (event == cv::EVENT_LBUTTONDOWN) {
                ui->handleMouseClick(x, y, 0.5);
            }
            else if (event == cv::EVENT_RBUTTONDOWN){
                ui->handleMouseClick(x, y, 2.0);
            }
            else if (event == cv::EVENT_MOUSEMOVE) {
                ui->handleMouseMove(x, y);
            }
        }
        
        void setData(int width, int height, double xMin, double xMax, double yMin, double yMax)
        {
            width_ = width;
            height_ = height;
            xMin_ = xMin;
            xMax_ = xMax;
            yMin_ = yMin;
            yMax_ = yMax;
        }

        void start() {
            
            mandelbrotSet_.calculateMandelbrotSetThread();            
            imageGenerator_.generateImage(mandelbrotSet_.getMandelbrotData());
            imageGenerator_.displayImage();
            //cv::waitKey(0);

            while (true)
            {
                int key = cv::waitKey(0);

                if (key == 'q' || key == 'Q')
                    break;

                if (key == 'w' || key == 'W')  // Up arrow key
                {
                    mandelbrotSet_.calculateMandelbrotSetThread();
                    imageGenerator_.generateImage(mandelbrotSet_.getMandelbrotData());
                    imageGenerator_.displayImage();
                }
            }
        }

        void handleMouseClick(int x, int y,double Factor) {

            double real = mapToRange(x, 0, width_ - 1, xMin_, xMax_);
            double imaginary = mapToRange(y, 0, height_ - 1, yMin_, yMax_);

            double zoomFactor = Factor;  // Adjust the zoom factor as needed

            double newWidth = (xMax_ - xMin_) * zoomFactor;
            double newHeight = (yMax_ - yMin_) * zoomFactor;
            double newCenterX = real;
            double newCenterY = imaginary;

            xMin_ = newCenterX - newWidth / 2.0;
            xMax_ = newCenterX + newWidth / 2.0;
            yMin_ = newCenterY - newHeight / 2.0;
            yMax_ = newCenterY + newHeight / 2.0;

            mandelbrotSet_.setData(800, 600, 1000, xMin_, xMax_, yMin_, yMax_);
            mandelbrotSet_.calculateMandelbrotSet1();
            imageGenerator_.generateImage(mandelbrotSet_.getMandelbrotData());
            imageGenerator_.displayImage();
        }

        void handleMouseMove(int x, int y) {
            // Handle mouse move event if needed
        }
        

private:
    int width_;
    int height_;
    double xMin_;
    double xMax_;
    double yMin_;
    double yMax_;
    MandelbrotSet mandelbrotSet_;
    ImageGenerator imageGenerator_;
    double mapToRange(int value, int inputMin, int inputMax, double outputMin, double outputMax) {
        return outputMin + ((value - inputMin) / static_cast<double>(inputMax - inputMin)) * (outputMax - outputMin);
    }
};


int main()
{
    const int width = 800;
    const int height = 600;
    const int maxIterations = 1000;
    const double xMin = -2.0;
    const double xMax = 1.0;
    const double yMin = -1.5;
    const double yMax = 1.5;

    MandelbrotSet mandelbrotSet(width, height, maxIterations, xMin, xMax, yMin, yMax);
    ImageGenerator imageGenerator(width, height);

    UserInterface userInterface(std::ref(mandelbrotSet), std::ref(imageGenerator));
    userInterface.setData(width, height, xMin, xMax, yMin, yMax);
    userInterface.start();



    return 0;
}
