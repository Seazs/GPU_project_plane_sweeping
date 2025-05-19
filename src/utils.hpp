#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdio>
#include <vector>
#include <fstream>
#include <cmath>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>


void save_cost_cube(const std::vector<cv::Mat>& cost_cube, const std::string& filename);

std::vector<cv::Mat> load_cost_cube(const std::string& filename);

bool compare_cost_cubes(const std::vector<cv::Mat>& cube1, const std::vector<cv::Mat>& cube2, float tol = 1e-4f);


#endif // UTILS_HPP